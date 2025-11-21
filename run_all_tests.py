
import subprocess
import sys
import os

def run_test(script_name):
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {script_name}")
    print(f"{'='*60}")
    
    # Use the same python interpreter as this script
    python_exe = sys.executable
    
    try:
        result = subprocess.run([python_exe, script_name], check=True)
        print(f"\n✅ {script_name} PASSED")
        return True
    except subprocess.CalledProcessError:
        print(f"\n❌ {script_name} FAILED")
        return False

def main():
    scripts = [
        "validate_system.py",
        "validate_advanced.py",
        "validate_integration.py"
    ]
    
    passed = 0
    failed = 0
    
    print("🧪 STARTING FULL SYSTEM VALIDATION SUITE")
    
    for script in scripts:
        if run_test(script):
            passed += 1
        else:
            failed += 1
            
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    print(f"Total Suites: {len(scripts)}")
    print(f"Passed:       {passed}")
    print(f"Failed:       {failed}")
    
    if failed == 0:
        print("\n✨ ALL SYSTEMS OPERATIONAL ✨")
        sys.exit(0)
    else:
        print("\n⚠️ SOME SYSTEMS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
