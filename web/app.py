import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import time
import altair as alt

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.evaluation.strength_evaluator import PasswordStrengthEvaluator
from src.simulator.password_cracker import PasswordCrackerSimulator
from evaluate_password import evaluate_password  # Import our trained model evaluator

# Set page config
st.set_page_config(
    page_title="Password Strength Evaluator",
    page_icon="🔐",
    layout="wide"
)

# Initialize services
@st.cache_resource
def load_evaluator_and_cracker():
    # Load dictionary for common passwords
    dict_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rockyou.txt")
    
    # Initialize common_passwords set
    common_passwords = set()
    if os.path.exists(dict_path):
        with open(dict_path, 'r', encoding='latin-1', errors='ignore') as f:
            # Load just the top passwords for efficiency
            for i, line in enumerate(f):
                if i >= 10000:  # Limit to top 10,000 passwords
                    break
                common_passwords.add(line.strip())
        st.sidebar.success(f"Loaded {len(common_passwords)} common passwords")
    else:
        st.sidebar.warning("Dictionary file not found. Some features will be limited.")
    
    # Initialize evaluator
    evaluator = PasswordStrengthEvaluator(common_passwords=common_passwords)
    
    # Initialize cracker simulator if dictionary is available
    cracker = None
    if os.path.exists(dict_path):
        cracker = PasswordCrackerSimulator(dictionary_path=dict_path)
    
    return evaluator, cracker

# Load services
evaluator, cracker = load_evaluator_and_cracker()

# Main app
st.title("AI-based Password Strength Evaluator")
st.markdown("Evaluate your password security and see how vulnerable it is to different attacks.")

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This app uses machine learning and password analysis techniques to "
    "evaluate password strength and simulate cracking attempts, providing "
    "feedback to improve password security."
)

# Password input
password = st.text_input("Enter a password to evaluate:", type="password")

# Create tabs
tab1, tab2 = st.tabs(["Strength Evaluation", "Attack Simulation"])

# Basic client-side strength indication
if password:
    # Simple visual strength indicator
    strength_score = 0
    
    # Criteria for visual display only 
    if len(password) > 0:
        # Length (up to 40 points)
        strength_score += min(40, len(password) * 3)
        
        # Character diversity (up to 60 points)
        if any(c.islower() for c in password): strength_score += 10
        if any(c.isupper() for c in password): strength_score += 15
        if any(c.isdigit() for c in password): strength_score += 15
        if any(not c.isalnum() for c in password): strength_score += 20
        
        # Simple pattern detection (subtract points)
        if any(pattern in password.lower() for pattern in ["password", "123", "qwerty", "admin"]): 
            strength_score = max(10, strength_score - 30)
    
    # Display strength meter
    if strength_score < 20:
        st.progress(strength_score/100, text="Very Weak")
        meter_color = "red"
    elif strength_score < 40:
        st.progress(strength_score/100, text="Weak")
        meter_color = "orange"
    elif strength_score < 60:
        st.progress(strength_score/100, text="Moderate")
        meter_color = "yellow"
    elif strength_score < 80:
        st.progress(strength_score/100, text="Strong")
        meter_color = "lightgreen"
    else:
        st.progress(strength_score/100, text="Very Strong")
        meter_color = "green"

# Tab 1: Strength Evaluation
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        evaluate_button = st.button("Evaluate Password Strength", use_container_width=True)
    
    with col2:
        generate_button = st.button("Generate Secure Password", use_container_width=True)
    
    if generate_button:
        import secrets
        import string
        
        def generate_password(length=16):
            alphabet = string.ascii_letters + string.digits + string.punctuation
            password = ''.join(secrets.choice(alphabet) for _ in range(length))
            return password
        
        new_password = generate_password()
        st.code(new_password, language=None)
        st.info("This is a cryptographically secure random password. Copy it and use a password manager to store it safely.")
    
    if evaluate_button and password:
        with st.spinner("Evaluating password strength..."):
            # Evaluate password using our trained models
            model_result = evaluate_password(password)
            
            # Display results
            st.subheader("AI Model Strength Assessment")
            
            # Main metrics with proper error handling
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                transformer_score = model_result.get('transformer_score', 0.0)
                st.metric("Transformer Score", f"{transformer_score:.2f}")
            
            with col2:
                lstm_score = model_result.get('lstm_score', 0.0)
                st.metric("LSTM Score", f"{lstm_score:.2f}")
            
            with col3:
                final_score = model_result.get('final_score', 0.0)
                st.metric("Final Score", f"{final_score:.2f}")
            
            with col4:
                transformer_confidence = model_result.get('transformer_confidence', 0.0)
                st.metric("Transformer Confidence", f"{transformer_confidence:.2f}")
            
            with col5:
                lstm_confidence = model_result.get('lstm_confidence', 0.0)
                st.metric("LSTM Confidence", f"{lstm_confidence:.2f}")
            
            # Display strength level with fallback
            st.subheader("Strength Level")
            strength_color = {
                "Very Weak": "red",
                "Weak": "orange",
                "Medium": "yellow",
                "Strong": "lightgreen",
                "Very Strong": "green"
            }
            strength_level = model_result.get('strength_level', 'Medium')
            st.markdown(f"<h3 style='color: {strength_color.get(strength_level, 'yellow')}'>{strength_level}</h3>", unsafe_allow_html=True)
            
            # Display model confidence visualization with fallbacks
            st.subheader("Model Confidence Analysis")
            
            # Create confidence factors DataFrame with fallback values
            confidence_factors = {
                'Factor': [
                    'Transformer Base',
                    'LSTM Base',
                    'Password Length',
                    'Character Types',
                    'Entropy',
                    'Pattern Analysis'
                ],
                'Value': [
                    transformer_score,
                    lstm_score,
                    min(1.0, model_result.get('analysis', {}).get('length', 8) / 16),
                    model_result.get('analysis', {}).get('character_types', 2) / 4,
                    min(1.0, model_result.get('analysis', {}).get('entropy', 40) / 80),
                    0.8 if model_result.get('analysis', {}).get('sequential_chars', False) or 
                         model_result.get('analysis', {}).get('repeated_chars', False) else 1.0
                ]
            }
            
            # Create confidence factors chart
            confidence_chart = alt.Chart(pd.DataFrame(confidence_factors)).mark_bar().encode(
                x='Factor',
                y='Value',
                color=alt.Color('Factor', scale=alt.Scale(range=['#4CAF50', '#2196F3', '#FFC107', '#FF5722', '#9C27B0', '#E91E63'])),
                tooltip=['Factor', 'Value']
            ).properties(
                title='Confidence Factors Breakdown',
                width=600,
                height=400
            )
            
            st.altair_chart(confidence_chart, use_container_width=True)
            
            # Display model comparison with fallbacks
            st.subheader("Model Comparison")
            
            # Create model comparison DataFrame with fallback values
            model_data = pd.DataFrame({
                'Model': ['Transformer', 'LSTM'],
                'Score': [transformer_score, lstm_score],
                'Confidence': [transformer_confidence, lstm_confidence]
            })
            
            # Create model comparison chart
            model_chart = alt.Chart(model_data).mark_bar().encode(
                x='Model',
                y='Score',
                color=alt.Color('Model', scale=alt.Scale(range=['#4CAF50', '#2196F3'])),
                tooltip=['Model', 'Score', 'Confidence']
            ).properties(
                title='Model Score and Confidence Comparison',
                width=600,
                height=400
            )
            
            # Add confidence bars
            confidence_chart = alt.Chart(model_data).mark_bar(
                opacity=0.3
            ).encode(
                x='Model',
                y='Confidence',
                color=alt.Color('Model', scale=alt.Scale(range=['#4CAF50', '#2196F3'])),
                tooltip=['Model', 'Score', 'Confidence']
            )
            
            st.altair_chart(model_chart + confidence_chart, use_container_width=True)
            
            # Display confidence explanation with fallbacks
            st.subheader("Confidence Analysis")
            
            if transformer_confidence < 0.5 or lstm_confidence < 0.5:
                st.warning("⚠️ Low confidence in evaluation. This may be due to:")
                st.markdown("""
                - Unusual password patterns
                - Insufficient training data for similar patterns
                - Conflicting model predictions
                """)
            elif transformer_confidence < 0.8 or lstm_confidence < 0.8:
                st.info("ℹ️ Moderate confidence in evaluation. Consider these factors:")
                st.markdown("""
                - Password characteristics are within normal ranges
                - Models show some agreement in predictions
                - Some uncertainty in pattern analysis
                """)
            else:
                st.success("✅ High confidence in evaluation. This means:")
                st.markdown("""
                - Password characteristics are well understood
                - Models strongly agree in their predictions
                - Pattern analysis is clear and consistent
                """)
            
            # Display detailed analysis with fallbacks
            st.subheader("Detailed Analysis")
            
            # Create analysis metrics with fallback values
            col1, col2, col3 = st.columns(3)
            
            with col1:
                length = model_result.get('analysis', {}).get('length', len(password) if password else 0)
                st.metric("Length", f"{length} characters")
            
            with col2:
                char_types = model_result.get('analysis', {}).get('character_types', 
                    len(set(c for c in password if c.isupper())) + 
                    len(set(c for c in password if c.islower())) + 
                    len(set(c for c in password if c.isdigit())) + 
                    len(set(c for c in password if not c.isalnum())) if password else 0)
                st.metric("Character Types", f"{char_types}/4")
            
            with col3:
                entropy = model_result.get('analysis', {}).get('entropy', 40)
                st.metric("Entropy", f"{entropy:.2f} bits")
            
            # Display recommendations with fallbacks
            st.subheader("Recommendations")
            recommendations = model_result.get('recommendations', [])
            if not recommendations:
                st.info("No specific recommendations available. Consider using a password manager and enabling two-factor authentication.")
            else:
                for rec in recommendations:
                    if rec.get('type') == 'critical':
                        st.error(rec.get('message', 'Critical security issue detected'))
                    elif rec.get('type') == 'warning':
                        st.warning(rec.get('message', 'Security warning'))
                    else:
                        st.info(rec.get('message', 'General recommendation'))

# Tab 2: Attack Simulation
with tab2:
    if not cracker:
        st.warning("Dictionary file not found. Attack simulation is not available.")
    else:
        simulate_button = st.button("Simulate Password Attacks", use_container_width=True)
        
        if simulate_button and password:
            with st.spinner("Simulating password attacks..."):
                # Add a small delay to make it feel like it's doing work
                max_attempts = 50000  # Limit for faster simulation
                
                # Run attack simulations
                with st.status("Running attack simulations...") as status:
                    st.write("Preparing simulations...")
                    time.sleep(0.5)
                    
                    st.write("Running brute force attack...")
                    time.sleep(0.5)
                    
                    st.write("Running dictionary attack...")
                    time.sleep(0.5)
                    
                    st.write("Running rule-based attack...")
                    time.sleep(0.5)
                    
                    # Actually run the attacks
                    results = cracker.run_all_attacks(
                        password, 
                        max_attempts_per_attack=max_attempts,
                        verbose=False
                    )
                    
                    status.update(label="Simulations complete!", state="complete")
                
                # Display theoretical vs simulated results
                st.subheader("Attack Simulation Results")
                
                # Prepare data for visualization with enhanced error handling
                attack_data = []
                total_successful_attacks = 0
                total_attempts = 0
                total_time = 0
                
                for attack_type, result in results['attacks'].items():
                    if 'error' not in result:
                        # Format attack type name
                        formatted_attack_type = ' '.join(word.capitalize() for word in attack_type.split('_'))
                        
                        # Calculate theoretical time based on password entropy
                        entropy = model_result.get('analysis', {}).get('entropy', 40) if 'model_result' in locals() else 40
                        
                        # Handle cases where attempts_per_second might be zero
                        attempts_per_second = result.get('attempts_per_second', 1)
                        if attempts_per_second <= 0:
                            attempts_per_second = 1  # Use a minimum value to avoid division by zero
                            
                        theoretical_time = 2 ** entropy / attempts_per_second
                        
                        # Calculate actual time and attempts per second with proper error handling
                        elapsed_time = result.get('elapsed_time', 0)
                        attempts = result.get('attempts', 0)
                        success = result.get('success', False)
                        
                        if elapsed_time > 0:
                            actual_attempts_per_second = attempts / elapsed_time
                        else:
                            actual_attempts_per_second = 0
                        
                        # Update totals
                        total_attempts += attempts
                        total_time += elapsed_time
                        if success:
                            total_successful_attacks += 1
                        
                        attack_data.append({
                            "Attack Type": formatted_attack_type,
                            "Attempts": attempts,
                            "Time (seconds)": round(elapsed_time, 2),
                            "Theoretical Time": round(theoretical_time, 2),
                            "Success": "Yes" if success else "No",
                            "Attempts per Second": round(actual_attempts_per_second, 2),
                            "Password Length": len(password) if password else 0,
                            "Character Types": len(set(c for c in password if c.isupper())) + 
                                             len(set(c for c in password if c.islower())) + 
                                             len(set(c for c in password if c.isdigit())) + 
                                             len(set(c for c in password if not c.isalnum())) if password else 0,
                            "Attack Speed": "Fast" if elapsed_time < 60 else "Medium" if elapsed_time < 300 else "Slow",
                            "Complexity": "High" if theoretical_time > 3600 else "Medium" if theoretical_time > 60 else "Low"
                        })
                
                # Create a DataFrame for visualization
                df = pd.DataFrame(attack_data)
                
                # Display enhanced metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Attempts", f"{total_attempts:,}", 
                            f"{total_successful_attacks} successful attacks")
                
                with col2:
                    success_rate = (total_successful_attacks / len(attack_data)) * 100 if attack_data else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%", 
                            "Higher is worse for security")
                
                with col3:
                    # Enhanced average speed calculation
                    if total_time > 0:
                        avg_speed = total_attempts / total_time
                        speed_label = f"{avg_speed:,.0f} attempts/sec"
                        if avg_speed > 1000000:
                            speed_label = f"{avg_speed/1000000:.1f}M attempts/sec"
                        elif avg_speed > 1000:
                            speed_label = f"{avg_speed/1000:.1f}K attempts/sec"
                    else:
                        avg_speed = 0
                        speed_label = "0 attempts/sec"
                    
                    # Add speed classification
                    speed_class = "Very Fast" if avg_speed > 1000000 else \
                                "Fast" if avg_speed > 100000 else \
                                "Moderate" if avg_speed > 10000 else \
                                "Slow" if avg_speed > 1000 else "Very Slow"
                    
                    st.metric("Average Speed", speed_label, 
                            f"{speed_class} - Higher is worse for security")
                
                with col4:
                    # Enhanced longest attack calculation
                    if not df.empty:
                        max_time = df['Time (seconds)'].max()
                        if max_time > 3600:  # More than 1 hour
                            time_label = f"{max_time/3600:.1f} hours"
                        elif max_time > 60:  # More than 1 minute
                            time_label = f"{max_time/60:.1f} minutes"
                        else:
                            time_label = f"{max_time:.1f} seconds"
                        
                        # Add time classification
                        time_class = "Very Long" if max_time > 3600 else \
                                   "Long" if max_time > 300 else \
                                   "Moderate" if max_time > 60 else \
                                   "Short" if max_time > 10 else "Very Short"
                    else:
                        time_label = "N/A"
                        time_class = "No attacks"
                    
                    st.metric("Longest Attack", time_label, 
                            f"{time_class} - Higher is better for security")
                
                # Create enhanced tabs for different visualizations
                viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
                    "Attack Performance", 
                    "Time Analysis", 
                    "Success Rate",
                    "Theoretical vs Actual",
                    "Attack Complexity"
                ])
                
                with viz_tab1:
                    # Enhanced Attack Performance Chart
                    chart = alt.Chart(df).mark_bar().encode(
                        x='Attack Type',
                        y='Attempts per Second',
                        color=alt.Color('Success', scale=alt.Scale(domain=['Yes', 'No'], range=['#4CAF50', '#F44336'])),
                        tooltip=['Attack Type', 'Attempts per Second', 'Success', 'Attempts', 'Time (seconds)']
                    ).properties(
                        title='Attack Performance (Attempts per Second)',
                        width=600,
                        height=400
                    )
                    st.altair_chart(chart, use_container_width=True)
                
                with viz_tab2:
                    # Enhanced Time Analysis Chart
                    time_chart = alt.Chart(df).mark_bar().encode(
                        x='Attack Type',
                        y='Time (seconds)',
                        color=alt.Color('Attack Speed', scale=alt.Scale(domain=['Fast', 'Medium', 'Slow'], 
                                                                       range=['#F44336', '#FFC107', '#4CAF50'])),
                        tooltip=['Attack Type', 'Time (seconds)', 'Success', 'Attempts', 'Attack Speed']
                    ).properties(
                        title='Time to Complete Each Attack',
                        width=600,
                        height=400
                    )
                    st.altair_chart(time_chart, use_container_width=True)
                
                with viz_tab3:
                    # Enhanced Success Rate Chart
                    success_chart = alt.Chart(df).mark_bar().encode(
                        x='Attack Type',
                        y=alt.Y('Success', aggregate='count'),
                        color='Success',
                        tooltip=['Attack Type', 'Success', 'Attempts', 'Time (seconds)']
                    ).properties(
                        title='Attack Success Rate',
                        width=600,
                        height=400
                    )
                    st.altair_chart(success_chart, use_container_width=True)
                
                with viz_tab4:
                    # Enhanced Theoretical vs Actual Time Chart
                    theoretical_chart = alt.Chart(df).mark_bar().encode(
                        x='Attack Type',
                        y='Theoretical Time',
                        color=alt.value('#FFC107'),
                        tooltip=['Attack Type', 'Theoretical Time', 'Complexity']
                    ).properties(
                        title='Theoretical vs Actual Attack Time',
                        width=600,
                        height=400
                    )
                    
                    actual_chart = alt.Chart(df).mark_bar().encode(
                        x='Attack Type',
                        y='Time (seconds)',
                        color=alt.value('#2196F3'),
                        tooltip=['Attack Type', 'Time (seconds)', 'Success']
                    )
                    
                    st.altair_chart(theoretical_chart + actual_chart, use_container_width=True)
                
                with viz_tab5:
                    # New Attack Complexity Chart
                    complexity_chart = alt.Chart(df).mark_bar().encode(
                        x='Attack Type',
                        y='Theoretical Time',
                        color=alt.Color('Complexity', scale=alt.Scale(domain=['Low', 'Medium', 'High'],
                                                                    range=['#F44336', '#FFC107', '#4CAF50'])),
                        tooltip=['Attack Type', 'Theoretical Time', 'Complexity', 'Success']
                    ).properties(
                        title='Attack Complexity Analysis',
                        width=600,
                        height=400
                    )
                    st.altair_chart(complexity_chart, use_container_width=True)
                
                # Display detailed results in an expandable section
                with st.expander("Detailed Attack Results"):
                    st.dataframe(df, use_container_width=True)
                
                # Enhanced security recommendations
                st.subheader("Security Recommendations")
                
                if total_successful_attacks > 0:
                    st.error("⚠️ Your password was cracked in the simulation! Consider these improvements:")
                    
                    # Specific recommendations based on attack type
                    successful_attacks = [attack for attack, result in results['attacks'].items() 
                                       if 'error' not in result and result.get('success', False)]
                    
                    if 'dictionary' in successful_attacks:
                        st.markdown("""
                        - Avoid using common words or phrases
                        - Don't use dictionary words without modification
                        - Consider using a passphrase with random words
                        - Use uncommon combinations of words
                        """)
                    
                    if 'brute_force' in successful_attacks:
                        st.markdown("""
                        - Increase password length (aim for 16+ characters)
                        - Use a mix of character types (uppercase, lowercase, numbers, symbols)
                        - Avoid predictable patterns
                        - Consider using a password manager
                        """)
                    
                    if 'pattern' in successful_attacks:
                        st.markdown("""
                        - Avoid keyboard patterns (qwerty, 12345)
                        - Don't use repeated characters
                        - Avoid common substitutions (e.g., @ for a)
                        - Use random character placement
                        """)
                    
                    st.markdown("""
                    **General Recommendations:**
                    - Use a password manager to generate and store secure passwords
                    - Enable two-factor authentication where available
                    - Use unique passwords for each account
                    - Regularly update your passwords
                    - Consider using a passphrase with special characters
                    """)
                else:
                    st.success("✅ Your password resisted all simulated attacks! Keep these best practices in mind:")
                    st.markdown("""
                    - Continue using strong, unique passwords
                    - Enable two-factor authentication where available
                    - Regularly update your passwords
                    - Use a password manager to maintain security
                    - Monitor your accounts for suspicious activity
                    - Consider using a passphrase for easier memorization
                    """)
                
                # Enhanced security score calculation
                st.subheader("Security Score")
                
                # Calculate security score based on various factors
                security_score = 100
                
                # Penalties for successful attacks
                if total_successful_attacks > 0:
                    security_score -= 50
                
                # Enhanced penalties for fast cracking attempts
                if avg_speed > 1000000:  # Very fast attacks
                    security_score -= 30
                elif avg_speed > 100000:  # Fast attacks
                    security_score -= 20
                elif avg_speed > 10000:  # Moderate speed
                    security_score -= 10
                
                # Enhanced time-based penalties
                if max_time < 60:  # Cracked in less than a minute
                    security_score -= 20
                elif max_time < 300:  # Cracked in less than 5 minutes
                    security_score -= 10
                
                # Bonuses for strong characteristics
                password_length = len(password) if password else 0
                character_types = len(set(c for c in password if c.isupper())) + \
                                len(set(c for c in password if c.islower())) + \
                                len(set(c for c in password if c.isdigit())) + \
                                len(set(c for c in password if not c.isalnum())) if password else 0
                
                if password_length >= 16:
                    security_score += 10
                elif password_length >= 12:
                    security_score += 5
                
                if character_types == 4:
                    security_score += 10
                elif character_types == 3:
                    security_score += 5
                
                if entropy >= 80:
                    security_score += 10
                elif entropy >= 60:
                    security_score += 5
                
                # Ensure score is between 0 and 100
                security_score = max(0, min(100, security_score))
                
                # Display security score with enhanced color coding
                score_color = {
                    "Very Weak": "red",
                    "Weak": "orange",
                    "Medium": "yellow",
                    "Strong": "lightgreen",
                    "Very Strong": "green"
                }
                
                if security_score < 40:
                    score_level = "Very Weak"
                elif security_score < 60:
                    score_level = "Weak"
                elif security_score < 80:
                    score_level = "Medium"
                elif security_score < 90:
                    score_level = "Strong"
                else:
                    score_level = "Very Strong"
                
                st.markdown(f"<h3 style='color: {score_color[score_level]}'>Security Score: {security_score}/100 ({score_level})</h3>", unsafe_allow_html=True)
                
                # Display enhanced progress bar with explanation
                st.progress(security_score/100)
                st.markdown(f"""
                **Score Breakdown:**
                - Base Score: 100
                - Successful Attacks Penalty: -{50 if total_successful_attacks > 0 else 0}
                - Speed Penalty: -{30 if avg_speed > 1000000 else 20 if avg_speed > 100000 else 10 if avg_speed > 10000 else 0}
                - Time Penalty: -{20 if max_time < 60 else 10 if max_time < 300 else 0}
                - Length Bonus: +{10 if password_length >= 16 else 5 if password_length >= 12 else 0}
                - Character Types Bonus: +{10 if character_types == 4 else 5 if character_types == 3 else 0}
                - Entropy Bonus: +{10 if entropy >= 80 else 5 if entropy >= 60 else 0}
                """)

# Footer
st.markdown("---")
st.markdown(
    """
    **Disclaimer:** This tool is for educational purposes only. Never share your actual passwords with any online service.
    The strength evaluation provides an estimate based on common patterns and entropy calculations, but cannot guarantee absolute security.
    """
)

# Sidebar - Advanced Options
with st.sidebar:
    st.subheader("Advanced Options")
    
    if st.checkbox("Show Password Statistics"):
        if password:
            st.write("**Character Distribution:**")
            
            # Count character frequencies
            char_counts = {}
            for c in password:
                if c in char_counts:
                    char_counts[c] += 1
                else:
                    char_counts[c] = 1
            
            # Convert to DataFrame
            char_df = pd.DataFrame({
                'Character': list(char_counts.keys()),
                'Count': list(char_counts.values())
            })
            
            # Create bar chart
            char_chart = alt.Chart(char_df).mark_bar().encode(
                x='Character:N',
                y='Count:Q',
                tooltip=['Character', 'Count']
            ).properties(
                title='Character Frequency',
                width=200,
                height=200
            )
            
            st.altair_chart(char_chart, use_container_width=True) 