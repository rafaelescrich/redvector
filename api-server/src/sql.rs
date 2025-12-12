//! SQL query parser and executor for RedVector
//! Similar to Qdrant's SQL-like query interface

use sqlparser::ast::{Statement, Query, SelectItem, Expr, BinaryOperator};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use redis::Commands;
use std::sync::Arc;
use std::collections::HashMap;

pub struct SqlExecutor {
    redis_client: Arc<redis::Client>,
}

impl SqlExecutor {
    pub fn new(redis_client: Arc<redis::Client>) -> Self {
        Self { redis_client }
    }

    /// Execute a SQL-like query
    /// Supported syntax:
    ///   SELECT * FROM collection_name WHERE vector = '[0.1, 0.2, ...]' LIMIT 10
    ///   SELECT id, score FROM collection_name WHERE vector = '[0.1, 0.2, ...]' LIMIT 10
    pub fn execute(&self, sql: &str) -> Result<SqlResult, SqlError> {
        let dialect = GenericDialect {};
        
        let ast = Parser::parse_sql(&dialect, sql)
            .map_err(|e| SqlError::ParseError(e.to_string()))?;

        if ast.is_empty() {
            return Err(SqlError::ParseError("Empty query".to_string()));
        }

        match &ast[0] {
            Statement::Query(query) => self.execute_query(query),
            _ => Err(SqlError::Unsupported("Only SELECT queries are supported".to_string())),
        }
    }

    fn execute_query(&self, query: &Query) -> Result<SqlResult, SqlError> {
        // Parse FROM clause to get collection name
        let collection_name = match query.body.as_ref() {
            sqlparser::ast::SetExpr::Select(select) => {
                match &select.from[0].relation {
                    sqlparser::ast::TableFactor::Table { name, .. } => {
                        name.0.last().unwrap().value.clone()
                    }
                    _ => return Err(SqlError::Unsupported("Invalid FROM clause".to_string())),
                }
            }
            _ => return Err(SqlError::Unsupported("Only SELECT is supported".to_string())),
        };

        // Parse WHERE clause to extract vector query
        let query_vector = match query.body.as_ref() {
            sqlparser::ast::SetExpr::Select(select) => {
                if let Some(selection) = &select.selection {
                    self.extract_vector_from_where(selection)?
                } else {
                    return Err(SqlError::Unsupported("WHERE clause with vector is required".to_string()));
                }
            }
            _ => return Err(SqlError::Unsupported("Only SELECT is supported".to_string())),
        };

        // Parse LIMIT
        let limit = query.limit
            .as_ref()
            .and_then(|l| {
                if let Expr::Value(sqlparser::ast::Value::Number(n, _)) = l {
                    n.parse::<usize>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(10)
            .min(100);

        // Execute search
        let mut conn = self.redis_client.get_connection()
            .map_err(|e| SqlError::ExecutionError(e.to_string()))?;

        let query_str = query_vector.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");

        let result: Result<redis::Value, redis::RedisError> = redis::cmd("FT.SEARCH")
            .arg(&collection_name)
            .arg(&query_str)
            .arg("LIMIT")
            .arg("0")
            .arg(limit.to_string())
            .query(&mut conn);

        match result {
            Ok(redis::Value::Bulk(mut arr)) => {
                if arr.is_empty() {
                    return Ok(SqlResult {
                        columns: vec!["id".to_string(), "score".to_string()],
                        rows: vec![],
                    });
                }

                let _count = arr.remove(0);
                let mut rows = Vec::new();

                let mut i = 0;
                while i < arr.len() {
                    if let (Some(redis::Value::Data(doc_id_bytes)), Some(redis::Value::Data(score_bytes))) = 
                        (arr.get(i), arr.get(i + 2)) {
                        if let (Ok(doc_id_str), Ok(score_str)) = 
                            (String::from_utf8(doc_id_bytes.clone()), String::from_utf8(score_bytes.clone())) {
                            rows.push(vec![doc_id_str, score_str]);
                        }
                    }
                    i += 3;
                }

                Ok(SqlResult {
                    columns: vec!["id".to_string(), "score".to_string()],
                    rows,
                })
            }
            Ok(redis::Value::Bulk(_)) | Ok(_) => Ok(SqlResult {
                columns: vec!["id".to_string(), "score".to_string()],
                rows: vec![],
            }),
            Err(e) => Err(SqlError::ExecutionError(e.to_string())),
        }
    }

    fn extract_vector_from_where(&self, expr: &Expr) -> Result<Vec<f32>, SqlError> {
        match expr {
            Expr::BinaryOp { left: _, op, right } => {
                if *op == BinaryOperator::Eq {
                    // Try both single-quoted and double-quoted strings
                    let vector_str = match right.as_ref() {
                        Expr::Value(sqlparser::ast::Value::SingleQuotedString(s)) => s.clone(),
                        Expr::Value(sqlparser::ast::Value::DoubleQuotedString(s)) => s.clone(),
                        _ => return Err(SqlError::Unsupported("Vector must be a quoted string".to_string())),
                    };
                    
                    // Parse vector from string like "[0.1, 0.2, 0.3]" or "0.1, 0.2, 0.3"
                    let trimmed = vector_str.trim_matches(|c| c == '[' || c == ']' || c == '"' || c == '\'');
                    let values: Result<Vec<f32>, _> = trimmed
                        .split(',')
                        .map(|s| s.trim().parse::<f32>())
                        .collect();
                    
                    values.map_err(|e| SqlError::ParseError(format!("Invalid vector format: {}", e)))
                } else {
                    Err(SqlError::Unsupported("Only = operator is supported for vector search".to_string()))
                }
            }
            _ => Err(SqlError::Unsupported("Unsupported WHERE clause format".to_string())),
        }
    }
}

#[derive(Debug)]
pub struct SqlResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

#[derive(Debug)]
pub enum SqlError {
    ParseError(String),
    Unsupported(String),
    ExecutionError(String),
}

impl std::fmt::Display for SqlError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SqlError::ParseError(e) => write!(f, "Parse error: {}", e),
            SqlError::Unsupported(e) => write!(f, "Unsupported: {}", e),
            SqlError::ExecutionError(e) => write!(f, "Execution error: {}", e),
        }
    }
}

impl std::error::Error for SqlError {}

