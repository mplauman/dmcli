# Overview

This is a Rust repository and you are an expert Rust programmer. You write
idiomatic Rust code that follows all current best practices.

### The Zen of Rust

Simple is better than complex.
Clear is better than clever.
Safety isn't optional.
Memory is precious.
Async when needed.
Don't panic.
Test, test, test.

## Code Generation

Use idiomatic practices at all times. In particular:

- Implement the `Display` trait for anything that should be user visible.
- Do not implement anything "for future use".
- Use the Builder pattern when constructing complex objects
- Use objects in `std::time` for time. Do not use u64 timestamps.
- Use async best practices.

## Validation

Create unit tests for new code to validate the new functionality. Do not
create demos, examples, or integration tests unless specifically asked.

Run `cargo fmt --all` and `cargo clippy --all-targets --all-features` to
validate code format. Fix all errors *and* warnings.

## Error Handling

This repository includes its own error type. Use that for fallible operations,
converting external errors into the repository's type by implementing the
`From` trait.

Use the `?` operator whenever possible, unless the error is truly impossible.
Use `expect` with a suitable description if an error will never happen or is
impossible to recover from.

## Documentation

Provide code documentation for new methods and modules. Do not create markdown
(.md) files describing features, functionality, or changes unless specifically
asked.

If it seems appropriate to update the `README.md` or other non-code
documentation, ask before making those changes. In general try not to.
