# ARC-AGI Project

## Virtual Environment Setup

This project uses a Python virtual environment to manage dependencies.

### Activating the Virtual Environment

```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scriptsctivate
```

### Installed Packages

The following packages are installed in the virtual environment:
- numpy
- pandas
- scikit-learn
- matplotlib
- tensorflow
- torch
- transformers
- requests

### Adding New Packages

```bash
# Activate the virtual environment first
source venv/bin/activate

# Install a new package
pip install package-name

# Update requirements.txt
pip freeze > requirements.txt
```

### Deactivating the Virtual Environment

```bash
deactivate
```
