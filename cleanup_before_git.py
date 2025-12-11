#!/usr/bin/env python3
"""
Security Cleanup Script for XAU Trading Bot
-------------------------------------------
This script removes sensitive files before pushing to GitHub.

USAGE:
    python cleanup_before_git.py [--dry-run]

OPTIONS:
    --dry-run    Show what would be deleted without actually deleting
"""

import os
import sys
import argparse
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}[WARNING] {text}{Colors.RESET}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}[ERROR] {text}{Colors.RESET}")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}[OK] {text}{Colors.RESET}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.BLUE}[INFO] {text}{Colors.RESET}")


def get_file_size_mb(filepath):
    """Get file size in MB."""
    try:
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    except:
        return 0


def find_sensitive_files(root_dir):
    """
    Find all files that should be removed before git push.

    Returns:
        dict: Dictionary with categories of files to remove
    """
    root = Path(root_dir)
    sensitive_files = {
        'logs': [],
        'csv_data': [],
        'output_files': [],
        'pycache': [],
        'total_size_mb': 0
    }

    # Find log files (CRITICAL - contains account numbers)
    for log_file in root.rglob('*.log'):
        size_mb = get_file_size_mb(log_file)
        sensitive_files['logs'].append((str(log_file), size_mb))
        sensitive_files['total_size_mb'] += size_mb

    # Find CSV data files (LARGE - can regenerate)
    for csv_file in root.rglob('*.csv'):
        size_mb = get_file_size_mb(csv_file)
        sensitive_files['csv_data'].append((str(csv_file), size_mb))
        sensitive_files['total_size_mb'] += size_mb

    # Find output.txt files
    for output_file in root.rglob('output.txt'):
        size_mb = get_file_size_mb(output_file)
        sensitive_files['output_files'].append((str(output_file), size_mb))
        sensitive_files['total_size_mb'] += size_mb

    # Find __pycache__ directories
    for pycache_dir in root.rglob('__pycache__'):
        if pycache_dir.is_dir():
            # Calculate directory size
            dir_size_mb = sum(
                get_file_size_mb(f)
                for f in pycache_dir.rglob('*')
                if f.is_file()
            )
            sensitive_files['pycache'].append((str(pycache_dir), dir_size_mb))
            sensitive_files['total_size_mb'] += dir_size_mb

    return sensitive_files


def delete_files(files_dict, dry_run=False):
    """
    Delete files or show what would be deleted.

    Args:
        files_dict: Dictionary of files to delete
        dry_run: If True, only show what would be deleted
    """
    total_deleted = 0
    total_size_freed = 0

    # Delete log files
    print_header("LOG FILES (Contains MT5 Account Numbers)")
    if files_dict['logs']:
        for filepath, size_mb in files_dict['logs']:
            if dry_run:
                print_warning(f"Would delete: {filepath} ({size_mb:.2f} MB)")
            else:
                try:
                    os.remove(filepath)
                    print_success(f"Deleted: {filepath} ({size_mb:.2f} MB)")
                    total_deleted += 1
                    total_size_freed += size_mb
                except Exception as e:
                    print_error(f"Failed to delete {filepath}: {e}")
    else:
        print_info("No log files found.")

    # Delete CSV data files
    print_header("CSV DATA FILES (Can Regenerate from MT5)")
    if files_dict['csv_data']:
        print_warning(f"Found {len(files_dict['csv_data'])} CSV file(s)")
        print_info("These are market data files that can be regenerated.")

        if not dry_run:
            response = input(f"\n{Colors.YELLOW}Delete CSV files? (y/N): {Colors.RESET}")
            if response.lower() != 'y':
                print_info("Skipping CSV files.")
                files_dict['csv_data'] = []

        for filepath, size_mb in files_dict['csv_data']:
            if dry_run:
                print_warning(f"Would delete: {filepath} ({size_mb:.2f} MB)")
            else:
                try:
                    os.remove(filepath)
                    print_success(f"Deleted: {filepath} ({size_mb:.2f} MB)")
                    total_deleted += 1
                    total_size_freed += size_mb
                except Exception as e:
                    print_error(f"Failed to delete {filepath}: {e}")
    else:
        print_info("No CSV data files found.")

    # Delete output.txt files
    print_header("OUTPUT TEXT FILES")
    if files_dict['output_files']:
        for filepath, size_mb in files_dict['output_files']:
            if dry_run:
                print_warning(f"Would delete: {filepath} ({size_mb:.2f} MB)")
            else:
                try:
                    os.remove(filepath)
                    print_success(f"Deleted: {filepath} ({size_mb:.2f} MB)")
                    total_deleted += 1
                    total_size_freed += size_mb
                except Exception as e:
                    print_error(f"Failed to delete {filepath}: {e}")
    else:
        print_info("No output.txt files found.")

    # Delete __pycache__ directories
    print_header("PYTHON CACHE DIRECTORIES")
    if files_dict['pycache']:
        for dirpath, size_mb in files_dict['pycache']:
            if dry_run:
                print_warning(f"Would delete: {dirpath} ({size_mb:.2f} MB)")
            else:
                try:
                    import shutil
                    shutil.rmtree(dirpath)
                    print_success(f"Deleted: {dirpath} ({size_mb:.2f} MB)")
                    total_deleted += 1
                    total_size_freed += size_mb
                except Exception as e:
                    print_error(f"Failed to delete {dirpath}: {e}")
    else:
        print_info("No __pycache__ directories found.")

    return total_deleted, total_size_freed


def check_gitignore():
    """Check if .gitignore exists and is properly configured."""
    print_header("CHECKING .gitignore")

    if not os.path.exists('.gitignore'):
        print_error(".gitignore not found!")
        print_info("Creating .gitignore is CRITICAL before git push.")
        return False

    # Check for critical patterns
    with open('.gitignore', 'r') as f:
        gitignore_content = f.read()

    critical_patterns = ['*.log', '*.csv']
    missing_patterns = []

    for pattern in critical_patterns:
        if pattern not in gitignore_content:
            missing_patterns.append(pattern)

    if missing_patterns:
        print_warning(f"Missing critical patterns in .gitignore: {', '.join(missing_patterns)}")
        return False

    print_success(".gitignore exists and contains critical patterns.")
    return True


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(
        description='Security cleanup script for XAU Trading Bot'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    args = parser.parse_args()

    print_header("XAU TRADING BOT - SECURITY CLEANUP")

    if args.dry_run:
        print_warning("DRY RUN MODE - No files will be deleted")

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Check .gitignore
    gitignore_ok = check_gitignore()
    if not gitignore_ok:
        print_error("\n.gitignore is missing or incomplete!")
        print_info("Please create/update .gitignore before proceeding.")
        if not args.dry_run:
            sys.exit(1)

    # Find sensitive files
    print_header("SCANNING FOR SENSITIVE FILES")
    print_info("Searching for log files, CSV data, and Python cache...")

    sensitive_files = find_sensitive_files('.')

    total_files = (
        len(sensitive_files['logs']) +
        len(sensitive_files['csv_data']) +
        len(sensitive_files['output_files']) +
        len(sensitive_files['pycache'])
    )

    print_info(f"\nFound {total_files} items to clean up")
    print_info(f"Total size: {sensitive_files['total_size_mb']:.2f} MB\n")

    if total_files == 0:
        print_success("No sensitive files found. Repository is clean!")
        return

    # Show summary
    print(f"{Colors.BOLD}Summary:{Colors.RESET}")
    print(f"  Log files:        {len(sensitive_files['logs'])}")
    print(f"  CSV data files:   {len(sensitive_files['csv_data'])}")
    print(f"  Output files:     {len(sensitive_files['output_files'])}")
    print(f"  Python cache:     {len(sensitive_files['pycache'])}")
    print()

    # Confirm deletion
    if not args.dry_run:
        print_warning("\nThis will PERMANENTLY delete the files listed above!")
        response = input(f"\n{Colors.BOLD}Continue? (y/N): {Colors.RESET}")
        if response.lower() != 'y':
            print_info("Cleanup cancelled.")
            sys.exit(0)

    # Delete files
    total_deleted, total_size_freed = delete_files(sensitive_files, args.dry_run)

    # Final summary
    print_header("CLEANUP COMPLETE")

    if args.dry_run:
        print_info(f"DRY RUN: Would delete {total_deleted} items")
        print_info(f"DRY RUN: Would free {total_size_freed:.2f} MB")
        print_info("\nRun without --dry-run to actually delete files.")
    else:
        print_success(f"Deleted {total_deleted} items")
        print_success(f"Freed {total_size_freed:.2f} MB")
        print_info("\nRepository is now safe for git push!")

    print("\n" + Colors.BOLD + "Next steps:" + Colors.RESET)
    print("  1. Review staged files: git status")
    print("  2. Initialize repository: git init")
    print("  3. Add files: git add .")
    print("  4. Review what will be committed: git status")
    print("  5. Commit: git commit -m 'Initial commit'")
    print("  6. Add remote: git remote add origin <your-repo-url>")
    print("  7. Push: git push -u origin main")
    print()


if __name__ == '__main__':
    main()
