---
name: release
description: Release a new version of dmcli, including version bumping, quality checks, PR creation, and git tagging.
---

# Release Process

Release a new version of `dmcli`.

## 1. Run Pre-flight Checks

Ensure all checks pass before cutting a release:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

## 2. Bump Version

1. Decide the new version number using Semantic Versioning.
2. In the root `Cargo.toml`, update `[workspace.package]` `version`.
3. Refresh `Cargo.lock`:
   ```sh
   cargo check
   ```

## 3. Open a Pull Request

The `main` branch requires changes to go through a pull request:

```sh
git checkout -b bump-v<VERSION>
git commit -am "chore: bump version to <VERSION>"
git push -u origin bump-v<VERSION>
```

Open and merge the pull request into `main`.

## 4. Tag and Publish

Once merged, update your local `main` branch and push an annotated tag:

```sh
git checkout main
git pull origin main
git tag -a v<VERSION> -m "Release v<VERSION>"
git push origin v<VERSION>
```

Pushing the tag triggers the `build.yml` GitHub Actions workflow, which builds binaries for all target platforms and creates the GitHub release.
