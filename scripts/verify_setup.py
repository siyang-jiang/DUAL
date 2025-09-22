#!/usr/bin/env python3
"""
Simple verification script to check repository setup.
This script doesn't require external dependencies.
"""

import os
import sys

def check_structure():
    """Check if the repository structure is correct."""
    print("DUAL Repository Structure Verification")
    print("=" * 50)
    
    # Check main directories
    required_dirs = [
        'src/dual',
        'configs', 
        'scripts',
        'tests',
        'data',
        'docs',
        'examples'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} (missing)")
            missing_dirs.append(dir_path)
    
    # Check key files
    print("\nKey Files:")
    required_files = [
        'README.md',
        'LICENSE', 
        'requirements.txt',
        'setup.py',
        'pyproject.toml',
        'configs/config.yaml',
        'src/dual/__init__.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            missing_files.append(file_path)
    
    # Check Python package structure
    print("\nPython Package Structure:")
    package_dirs = [
        'src/dual',
        'src/dual/models', 
        'src/dual/data',
        'src/dual/utils'
    ]
    
    for pkg_dir in package_dirs:
        init_file = os.path.join(pkg_dir, '__init__.py')
        if os.path.exists(init_file):
            print(f"✓ {pkg_dir}/__init__.py")
        else:
            print(f"✗ {pkg_dir}/__init__.py (missing)")
    
    print("\n" + "=" * 50)
    if missing_dirs or missing_files:
        print(f"❌ Setup incomplete. Missing {len(missing_dirs + missing_files)} items.")
        return False
    else:
        print("✅ Repository structure is complete!")
        return True

def check_import():
    """Check if the package can be imported."""
    print("\nPackage Import Test:")
    print("-" * 30)
    
    try:
        sys.path.insert(0, 'src')
        import dual
        print("✓ dual package imported successfully")
        print(f"✓ Package version: {dual.__version__}")
        print(f"✓ Package author: {dual.__author__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import dual package: {e}")
        return False

def main():
    """Main verification function."""
    structure_ok = check_structure()
    import_ok = check_import()
    
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    if structure_ok and import_ok:
        print("🎉 All checks passed! Repository is ready for development.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Install in development mode: pip install -e .")
        print("3. Download datasets (see data/README.md)")
        print("4. Run training: python scripts/train.py")
        return 0
    else:
        print("❌ Some checks failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())