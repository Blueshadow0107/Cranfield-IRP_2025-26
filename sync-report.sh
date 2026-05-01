#!/bin/bash
# Report/ Directory Sync Tool
# Syncs Report/ directory with the report-only branch for Overleaf collaboration

set -e

show_help() {
    echo "==================================="
    echo "Report/ Directory Sync Tool"
    echo "==================================="
    echo ""
    echo "Usage: ./sync-report.sh [command]"
    echo ""
    echo "Commands:"
    echo "  push    - Push local Report/ changes to Overleaf (report-only branch)"
    echo "  pull    - Pull Overleaf changes to local Report/"
    echo "  status  - Check differences between local and Overleaf"
    echo "  help    - Show this help message"
    echo ""
}

# Check if we're in the right directory
if [ ! -d "Report" ]; then
    echo "Error: Report/ directory not found. Run this from the project root."
    exit 1
fi

case "${1:-help}" in
    push)
        echo "==================================="
        echo "Report/ Directory Sync Tool"
        echo "==================================="
        echo ""
        echo "Pushing local changes to report-only branch..."
        echo ""
        
        # Commit any pending changes first
        if [ -n "$(git status --porcelain Report/)" ]; then
            git add Report/
            git commit -m "Update Report from local" || true
        fi
        
        # Use subtree push
        git subtree push --prefix=Report origin report-only
        
        echo ""
        echo "✓ Successfully pushed to Overleaf (report-only branch)"
        echo ""
        ;;
        
    pull)
        echo "==================================="
        echo "Report/ Directory Sync Tool"
        echo "==================================="
        echo ""
        echo "Pulling changes from report-only branch..."
        echo ""
        
        # Fetch latest
        git fetch origin report-only
        
        # Check for differences
        if git diff --quiet HEAD origin/report-only -- Report/ 2>/dev/null; then
            # No Report directory in diff, check if content differs
            local_files=$(find Report/ -type f | wc -l)
            temp_dir=$(mktemp -d)
            git archive origin/report-only | tar -x -C "$temp_dir"
            remote_files=$(find "$temp_dir" -type f | wc -l)
            
            if [ "$local_files" -eq "$remote_files" ] && diff -rq Report/ "$temp_dir" >/dev/null 2>&1; then
                echo "✓ Report/ is already up to date with Overleaf"
                rm -rf "$temp_dir"
                exit 0
            fi
            rm -rf "$temp_dir"
        fi
        
        # Backup current Report
        backup_dir="Report_backup_$(date +%Y%m%d_%H%M%S)"
        cp -r Report/ "$backup_dir"
        
        # Extract report-only content
        rm -rf Report/
        mkdir Report
        temp_dir=$(mktemp -d)
        git archive origin/report-only | tar -x -C "$temp_dir"
        cp -r "$temp_dir"/* Report/
        rm -rf "$temp_dir"
        
        echo "✓ Successfully updated Report/ from Overleaf"
        echo "  (Backup saved to: $backup_dir)"
        echo ""
        
        # Show what changed
        if command -v git &> /dev/null; then
            git add Report/
            git status --short Report/ 2>/dev/null || true
        fi
        echo ""
        echo "Run 'git add Report/' and commit if you want to keep these changes."
        echo ""
        ;;
        
    status)
        echo "==================================="
        echo "Report/ Directory Sync Tool"
        echo "==================================="
        echo ""
        echo "Checking status..."
        echo ""
        
        # Fetch latest
        git fetch origin report-only 2>/dev/null || true
        
        # Create temp copy of remote
        temp_dir=$(mktemp -d)
        git archive origin/report-only 2>/dev/null | tar -x -C "$temp_dir" || true
        
        echo "Local Report/ files:"
        find Report/ -type f | wc -l
        echo ""
        echo "Remote (Overleaf) files:"
        find "$temp_dir" -type f 2>/dev/null | wc -l
        echo ""
        
        # Show differences
        if diff -rq Report/ "$temp_dir" 2>/dev/null; then
            echo "✓ Local and Overleaf are in sync"
        else
            echo "Differences found (files that differ or are only in one location):"
            diff -rq Report/ "$temp_dir" 2>/dev/null | grep -E "(differ|Only)" || echo "  (Unable to show detailed diff)"
        fi
        
        rm -rf "$temp_dir"
        echo ""
        ;;
        
    help|--help|-h)
        show_help
        ;;
        
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
