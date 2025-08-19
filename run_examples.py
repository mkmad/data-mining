#!/usr/bin/env python3
"""
Data Mining Examples Runner

This script helps you run different examples from the organized data mining projects.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header():
    """Print a nice header for the script."""
    print("=" * 60)
    print("🔬 Data Mining Examples Runner")
    print("=" * 60)
    print()

def print_menu():
    """Print the main menu options."""
    print("📚 Available Projects:")
    print()
    print("1. 🍽️  Meal Detection (Train Model)")
    print("2. 🍽️  Meal Detection (Make Predictions)")
    print("3. ⏰ Time Series Analysis")
    print("4. 🔗 Clustering Validation")
    print("5. 📖 View Documentation")
    print("6. 🚪 Exit")
    print()

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print(f"🚀 Running: {description}")
    print(f"📁 Script: {script_path}")
    print("-" * 40)
    
    try:
        # Change to the script's directory
        script_dir = os.path.dirname(script_path)
        if script_dir:
            os.chdir(script_dir)
        
        # Run the script
        result = subprocess.run([sys.executable, os.path.basename(script_path)], 
                              capture_output=True, text=True)
        
        if result.stdout:
            print("📤 Output:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️  Warnings/Errors:")
            print(result.stderr)
            
        print("-" * 40)
        print("✅ Script completed!")
        
    except Exception as e:
        print(f"❌ Error running script: {e}")
        print("-" * 40)
    
    # Return to original directory
    os.chdir(Path(__file__).parent)

def show_documentation():
    """Show available documentation."""
    print("📖 Available Documentation:")
    print()
    
    docs_dir = Path("documentation")
    if docs_dir.exists():
        for pdf_file in docs_dir.glob("*.pdf"):
            print(f"   📄 {pdf_file.name}")
    else:
        print("   ❌ Documentation directory not found")
    
    print()
    print("💡 Tip: Open these PDF files to learn about the projects")
    print()

def main():
    """Main function to run the interactive menu."""
    print_header()
    
    while True:
        print_menu()
        
        try:
            choice = input("🔬 Enter your choice (1-6): ").strip()
            print()
            
            if choice == "1":
                run_script("projects/meal-detection/train-model.py", "Meal Detection - Model Training")
                
            elif choice == "2":
                run_script("projects/meal-detection/predict.py", "Meal Detection - Making Predictions")
                
            elif choice == "3":
                run_script("projects/time-series-analysis/time-series-extraction.py", "Time Series Analysis")
                
            elif choice == "4":
                run_script("projects/clustering-validation/cluster_validation.py", "Clustering Validation")
                
            elif choice == "5":
                show_documentation()
                
            elif choice == "6":
                print("👋 Thanks for exploring the data mining projects!")
                print("📚 Check out the README files in each directory for more details.")
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-6.")
                print()
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print()

if __name__ == "__main__":
    main()
