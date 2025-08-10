#!/bin/bash

# Open GitHub Personal Access Token (classic) generation page in browser
echo "Opening GitHub token generation page..."
xdg-open "https://github.com/settings/tokens/new?scopes=repo&description=Git+Push+Token" >/dev/null 2>&1

echo "Please log in to GitHub (if not already), generate a token with 'repo' access, and copy it."
read -s -p "Paste your new GitHub token here: " TOKEN
echo

# Ask for GitHub username
read -p "Enter your GitHub username: " USERNAME

# Update git remote to include username
git remote set-url origin https://${USERNAME}@github.com/$(git remote get-url origin | sed -E 's#https://([^/]+/)?##')

# Store credentials
git config --global credential.helper store
cat <<EOF > ~/.git-credentials
https://${USERNAME}:${TOKEN}@github.com
EOF

# Push changes
git push
