# Password Strength Evaluator

An AI-powered password strength evaluation and attack simulation tool that helps users understand and improve their password security.

## Features

- **AI Model Strength Assessment**: Uses both Transformer and LSTM models to evaluate password strength
- **Attack Simulation**: Simulates various password cracking attempts
- **Detailed Analysis**: Provides comprehensive security metrics and recommendations
- **Visual Analytics**: Interactive charts and visualizations of attack results
- **Security Scoring**: Calculates a security score based on multiple factors

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/password-strength-evaluator.git
cd password-strength-evaluator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run web/app.py
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Streamlit
- Pandas
- NumPy
- Altair
- scikit-learn

## Project Structure

```
password-strength-evaluator/
├── data/               # Data files and dictionaries
├── models/            # Trained model files
├── src/              # Source code
│   ├── models/       # Model implementations
│   ├── evaluation/   # Evaluation modules
│   └── simulator/    # Attack simulation code
├── web/              # Web application
│   └── app.py        # Streamlit application
├── requirements.txt  # Python dependencies
└── README.md        # Project documentation
```

## Usage

1. Enter a password in the input field
2. View the AI model strength assessment
3. Run attack simulations to see how your password performs
4. Review detailed security metrics and recommendations
5. Use the insights to improve your password security

## Security Note

This tool is for educational purposes only. Never share your actual passwords with any online service. The strength evaluation provides an estimate based on common patterns and entropy calculations, but cannot guarantee absolute security.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request 