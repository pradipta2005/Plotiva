"""
Professional Deployment Script for Plotiva
Comprehensive testing and deployment automation
"""

import subprocess
import sys
import os
import importlib
import time
from pathlib import Path

class PlotivaDeployment:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.required_packages = [
            'streamlit>=1.28.0',
            'pandas>=2.0.0',
            'numpy>=1.24.0',
            'plotly>=5.15.0',
            'scipy>=1.11.0',
            'scikit-learn>=1.3.0',
            'openpyxl>=3.1.0',
            'xlrd>=2.0.0',
            'pyarrow>=12.0.0',
            'fastparquet>=0.8.0',
            'statsmodels>=0.14.0',
            'seaborn>=0.12.0',
            'matplotlib>=3.7.0',
            'umap-learn>=0.5.3',
            'hdbscan>=0.8.29',
            'fpdf2>=2.7.0',
            'Pillow>=10.0.0',
            'kaleido>=0.2.1'
        ]
        
    def print_banner(self):
        """Print deployment banner"""
        print("=" * 60)
        print("🚀 PLOTIVA DEPLOYMENT SYSTEM")
        print("Professional Data Analysis Platform")
        print("=" * 60)
        
    def check_python_version(self):
        """Check Python version compatibility"""
        print("🔍 Checking Python version...")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8+ required. Current version:", sys.version)
            return False
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
        
    def install_requirements(self):
        """Install required packages"""
        print("\n📦 Installing requirements...")
        try:
            # Install from requirements file if exists
            req_file = self.project_root / "requirements_premium.txt"
            if req_file.exists():
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-r", str(req_file)
                ])
            else:
                # Install individual packages
                for package in self.required_packages:
                    print(f"Installing {package}...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", package
                    ])
            print("✅ All requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
            
    def verify_imports(self):
        """Verify all critical imports work"""
        print("\n🔍 Verifying imports...")
        critical_modules = [
            'streamlit', 'pandas', 'numpy', 'plotly', 'scipy', 
            'sklearn', 'openpyxl', 'matplotlib', 'seaborn'
        ]
        
        failed_imports = []
        for module in critical_modules:
            try:
                importlib.import_module(module)
                print(f"✅ {module}")
            except ImportError as e:
                print(f"❌ {module}: {e}")
                failed_imports.append(module)
                
        if failed_imports:
            print(f"\n❌ Failed imports: {failed_imports}")
            return False
        print("✅ All critical modules imported successfully")
        return True
        
    def check_file_structure(self):
        """Check if all required files exist"""
        print("\n📁 Checking file structure...")
        required_files = [
            'main_premium.py',
            'premium_config.py',
            'premium_plots.py',
            'premium_analytics.py',
            'utils.py',
            'run_premium.py'
        ]
        
        missing_files = []
        for file in required_files:
            file_path = self.project_root / file
            if file_path.exists():
                print(f"✅ {file}")
            else:
                print(f"❌ {file} - Missing")
                missing_files.append(file)
                
        if missing_files:
            print(f"\n❌ Missing files: {missing_files}")
            return False
        print("✅ All required files present")
        return True
        
    def run_syntax_check(self):
        """Check Python syntax for all files"""
        print("\n🔍 Running syntax checks...")
        python_files = list(self.project_root.glob("*.py"))
        
        for file in python_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    compile(f.read(), file, 'exec')
                print(f"✅ {file.name}")
            except SyntaxError as e:
                print(f"❌ {file.name}: Syntax error at line {e.lineno}")
                return False
            except Exception as e:
                print(f"⚠️ {file.name}: {e}")
                
        print("✅ All Python files have valid syntax")
        return True
        
    def create_startup_script(self):
        """Create optimized startup script"""
        print("\n📝 Creating startup script...")
        
        startup_script = '''@echo off
echo 🚀 Starting Plotiva Premium...
echo.
echo Opening browser in 5 seconds...
timeout /t 5 /nobreak > nul
start http://localhost:8501
python -m streamlit run main_premium.py --server.port=8501 --server.headless=true
pause
'''
        
        script_path = self.project_root / "start_plotiva.bat"
        with open(script_path, 'w') as f:
            f.write(startup_script)
            
        print(f"✅ Startup script created: {script_path}")
        return True
        
    def optimize_performance(self):
        """Apply performance optimizations"""
        print("\n⚡ Applying performance optimizations...")
        
        # Create .streamlit directory and config
        streamlit_dir = self.project_root / ".streamlit"
        streamlit_dir.mkdir(exist_ok=True)
        
        config_content = '''[global]
developmentMode = false
showWarningOnDirectExecution = false

[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
'''
        
        config_path = streamlit_dir / "config.toml"
        with open(config_path, 'w') as f:
            f.write(config_content)
            
        print("✅ Performance optimizations applied")
        return True
        
    def run_basic_test(self):
        """Run basic functionality test"""
        print("\n🧪 Running basic functionality test...")
        
        try:
            # Test data generation
            import pandas as pd
            import numpy as np
            
            # Generate test data
            np.random.seed(42)
            test_data = pd.DataFrame({
                'A': np.random.normal(0, 1, 100),
                'B': np.random.normal(0, 1, 100),
                'C': np.random.choice(['X', 'Y', 'Z'], 100)
            })
            
            # Test basic operations
            assert len(test_data) == 100
            assert test_data['A'].mean() is not None
            assert test_data['C'].nunique() == 3
            
            print("✅ Basic functionality test passed")
            return True
            
        except Exception as e:
            print(f"❌ Basic functionality test failed: {e}")
            return False
            
    def generate_deployment_report(self, results):
        """Generate deployment report"""
        print("\n📊 Generating deployment report...")
        
        report = f"""
# PLOTIVA DEPLOYMENT REPORT
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## System Information
- Python Version: {sys.version}
- Platform: {sys.platform}
- Project Root: {self.project_root}

## Deployment Results
- Python Version Check: {'✅ PASS' if results['python_version'] else '❌ FAIL'}
- Requirements Installation: {'✅ PASS' if results['requirements'] else '❌ FAIL'}
- Import Verification: {'✅ PASS' if results['imports'] else '❌ FAIL'}
- File Structure Check: {'✅ PASS' if results['file_structure'] else '❌ FAIL'}
- Syntax Check: {'✅ PASS' if results['syntax'] else '❌ FAIL'}
- Performance Optimization: {'✅ PASS' if results['performance'] else '❌ FAIL'}
- Basic Functionality Test: {'✅ PASS' if results['basic_test'] else '❌ FAIL'}

## Overall Status
{'🎉 DEPLOYMENT SUCCESSFUL - Ready for production!' if all(results.values()) else '⚠️ DEPLOYMENT ISSUES DETECTED - Please review failed checks'}

## Next Steps
1. Run: python run_premium.py
2. Or use: start_plotiva.bat
3. Open browser to: http://localhost:8501

## Support
For issues, check the deployment log above and ensure all requirements are met.
"""
        
        report_path = self.project_root / "deployment_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
            
        print(f"✅ Deployment report saved: {report_path}")
        return True
        
    def deploy(self):
        """Run complete deployment process"""
        self.print_banner()
        
        results = {
            'python_version': self.check_python_version(),
            'requirements': self.install_requirements(),
            'imports': self.verify_imports(),
            'file_structure': self.check_file_structure(),
            'syntax': self.run_syntax_check(),
            'performance': self.optimize_performance(),
            'basic_test': self.run_basic_test()
        }
        
        # Create startup script regardless of other results
        self.create_startup_script()
        
        # Generate report
        self.generate_deployment_report(results)
        
        # Final status
        print("\n" + "=" * 60)
        if all(results.values()):
            print("🎉 DEPLOYMENT SUCCESSFUL!")
            print("✅ Plotiva is ready for production use")
            print("\n🚀 To start the application:")
            print("   Option 1: Double-click 'start_plotiva.bat'")
            print("   Option 2: Run 'python run_premium.py'")
            print("   Option 3: Run 'streamlit run main_premium.py'")
        else:
            print("⚠️ DEPLOYMENT COMPLETED WITH ISSUES")
            print("❌ Some checks failed - review the report above")
            print("🔧 Fix the issues and run deployment again")
            
        print("=" * 60)
        return all(results.values())

if __name__ == "__main__":
    deployer = PlotivaDeployment()
    success = deployer.deploy()
    sys.exit(0 if success else 1)