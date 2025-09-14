"""
Premium Data Analysis Platform Launcher
Run this file to start the premium application
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_premium.txt"])
        print("✅ All requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def run_app():
    """Run the premium Streamlit application"""
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main_premium.py", "--server.port=8501"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    print("🚀 Premium Data Analysis Platform")
    print("=" * 50)
    
    # Check if requirements file exists
    if not os.path.exists("requirements_premium.txt"):
        print("❌ requirements_premium.txt not found!")
        sys.exit(1)
    
    # Install requirements
    print("📦 Installing requirements...")
    if not install_requirements():
        sys.exit(1)
    
    print("\n🎉 Starting Premium Data Analysis Platform...")
    print("🌐 Open your browser to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)
    
    # Run the application
    run_app()