final_version_code_repo

Purpose
- Code-only mirror of key training-related modules from final_version.
- Includes: models/, training/, utils/ (Python only).
- Excludes data, checkpoints, logs, and non-code assets.

Update
- To refresh from source tree:
  rsync -a --prune-empty-dirs \
    --include="models/" --include="models/**/" --include="models/**/*.py" \
    --include="training/" --include="training/**/" --include="training/**/*.py" \
    --include="utils/" --include="utils/**/" --include="utils/**/*.py" \
    --exclude="*" ../final_version/ ./

Remote
- After init/commit, add your remote and push:
  git remote add origin <YOUR_REMOTE_URL>
  git branch -M main
  git push -u origin main
