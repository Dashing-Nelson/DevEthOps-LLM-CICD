#!/usr/bin/env python3
"""
Script to fix Unicode characters in logging for Windows compatibility.
"""

import os
import re
from pathlib import Path

def fix_unicode_in_file(filepath):
    """Replace Unicode emojis with ASCII equivalents."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace Unicode emojis with ASCII equivalents
        replacements = {
            '✅': '[PASS]',
            '❌': '[FAIL]', 
            '🎉': '[SUCCESS]',
            '🚀': '[START]',
            '⚖️': '[FAIRNESS]',
            '🔍': '[CHECKING]',
            '🔧': '[PROCESSING]',
            '🤖': '[MODEL]'
        }
        
        modified = False
        for emoji, replacement in replacements.items():
            if emoji in content:
                content = content.replace(emoji, replacement)
                modified = True
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed Unicode characters in: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Fix Unicode characters in all Python files."""
    src_dir = Path('src')
    
    files_fixed = 0
    
    for py_file in src_dir.rglob('*.py'):
        if fix_unicode_in_file(py_file):
            files_fixed += 1
    
    # Also fix test files
    for py_file in Path('.').glob('test_*.py'):
        if fix_unicode_in_file(py_file):
            files_fixed += 1
            
    print(f"Fixed {files_fixed} files with Unicode issues")

if __name__ == "__main__":
    main()
