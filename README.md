# CyberAI Password Strength Evaluator

An advanced AI-powered password strength evaluation and attack simulation tool that helps users understand and improve their password security through deep learning models and comprehensive attack simulations.

## Features

- **AI-Powered Strength Assessment**
  - Transformer model for complex pattern recognition
  - LSTM model for sequential pattern analysis
  - Combined model evaluation for robust strength assessment

- **Advanced Attack Simulation**
  - Dictionary-based attacks
  - Brute force attempts
  - Pattern-based cracking simulations
  - Real-time attack progress monitoring

- **Comprehensive Security Analysis**
  - Detailed password characteristics evaluation
  - Multiple security metrics calculation
  - Theoretical vs. actual cracking time comparison
  - Password entropy assessment

- **Interactive Visualization**
  - Real-time attack progress charts
  - Security score breakdown
  - Password strength comparisons
  - Attack success rate visualizations

- **Smart Recommendations**
  - Context-aware security suggestions
  - Pattern vulnerability identification
  - Specific improvement strategies
  - Best practices guidance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cyberai.git
cd cyberai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
cyberai/
├── src/                    # Source code
│   ├── api/               # API implementations
│   ├── evaluation/        # Password evaluation logic
│   ├── models/           # AI model implementations
│   ├── preprocessing/    # Data processing utilities
│   ├── simulator/        # Attack simulation modules
│   └── utils/           # Helper utilities
├── web/                  # Web interface
│   ├── app.py           # Main web application
│   └── index.html       # Frontend template
├── evaluate_password.py  # Password evaluation script
├── train_models.py      # Model training script
├── run.py               # Application entry point
└── requirements.txt     # Project dependencies
```

## Usage

1. Start the web application:
```bash
python run.py
```

2. Access the interface at `http://localhost:5000`

3. Enter a password to:
   - Get AI-powered strength assessment
   - View attack simulation results
   - Receive detailed security recommendations
   - See visualization of security metrics

## Security Features

### Password Strength Evaluation
- Character composition analysis
- Pattern recognition
- Entropy calculation
- Common vulnerability detection

### Attack Simulations
- Dictionary-based attacks using common wordlists
- Brute force attempts with various character sets
- Pattern-based attacks targeting common substitutions
- Time-based analysis of cracking attempts

### Security Metrics
- Overall security score (0-100)
- Theoretical cracking time
- Actual simulation results
- Comparative strength analysis

## Development

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Flask
- NumPy
- Pandas
- scikit-learn

### Setting up Development Environment
1. Fork the repository
2. Create a new branch for your feature
3. Install development dependencies
4. Make your changes
5. Submit a pull request

## Security Notice

This tool is designed for educational and security assessment purposes only. Never input actual passwords used for your accounts. The evaluations and simulations provide estimates based on known patterns and attack methods but cannot guarantee absolute security.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the deep learning framework
- Security researchers and the cybersecurity community
- Contributors and testers