# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for RedVector.

## What is an ADR?

An Architecture Decision Record captures an important architectural decision made along with its context and consequences.

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](./ADR-001-GPU-ACCELERATION.md) | GPU Acceleration for Vector Search | Proposed | 2024-12-23 |
| [ADR-002](./ADR-002-ARCHITECTURE-ADVANTAGES.md) | Architecture Advantages & Vector Storage | Proposed | 2024-12-24 |
| [ADR-003](./ADR-003-RVF-V2-MULTIVECTOR.md) | RVF v2: Multi-Vector Storage Format | Proposed | 2025-12-25 |

## ADR Status Definitions

- **Proposed** - Under discussion, not yet approved
- **Accepted** - Approved for implementation
- **Deprecated** - No longer relevant or superseded
- **Superseded** - Replaced by another ADR

## Creating New ADRs

Use the following naming convention: `ADR-XXX-TITLE.md`

Template:
```markdown
# ADR-XXX: Title

| Status | Proposed |
|--------|----------|
| **Date** | YYYY-MM-DD |
| **Decision Makers** | Names |
| **Technical Area** | Area |

## Context and Problem Statement
...

## Decision Drivers
...

## Considered Options
...

## Decision Outcome
...

## Consequences
...
```

