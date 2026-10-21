<!-- Thanks for contributing! See CONTRIBUTING.md for the full guide. -->

## What this changes

<!-- What the change does and why. Link the issue it addresses, if any. -->

## How it was tested

<!-- Which tests you added or ran, and on which operating system. Screenshots for anything visible. -->

## Checklist

- [ ] `cargo fmt --all`, `cargo clippy --workspace --all-targets` and `cargo test --workspace` pass
- [ ] New behaviour has tests (blocks and execution changes: a scenario in `crates/magi-engine/tests/engines.rs`)
- [ ] Both engines were updated if block behaviour changed
- [ ] `docs/NODES.md` was regenerated if a block definition changed
- [ ] `docs/FILE_FORMAT.md` was updated if a file format changed, and old files still load
- [ ] `CHANGELOG.md` has an entry if users would notice the change
- [ ] I agree to the contribution terms in [EULA.md](../EULA.md)
