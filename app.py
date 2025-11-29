import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from io import StringIO

# Page configuration
st.set_page_config(page_title="Distribution Fitting Tool", layout="wide", page_icon="📊")

# Title and description
st.title("📊 Statistical Distribution Fitting Tool")
st.markdown("Fit various probability distributions to your data and visualize the results.")

# Available distributions with their scipy.stats objects
DISTRIBUTIONS = {
    'Normal': stats.norm,
    'Gamma': stats.gamma,
    'Weibull': stats.weibull_min,
    'Exponential': stats.expon,
    'Log-Normal': stats.lognorm,
    'Beta': stats.beta,
    'Chi-Square': stats.chi2,
    'Uniform': stats.uniform,
    'Rayleigh': stats.rayleigh,
    'Pareto': stats.pareto,
    'Laplace': stats.laplace,
    'Cauchy': stats.cauchy
}

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'manual_mode' not in st.session_state:
    st.session_state.manual_mode = False

# Sidebar for data input
with st.sidebar:
    st.header("📥 Data Input")
    
    input_method = st.radio("Choose input method:", ["Manual Entry", "Upload CSV"])
    
    if input_method == "Manual Entry":
        st.markdown("**Enter your data** (comma or space separated):")
        data_input = st.text_area("Data:", height=150, 
                                   placeholder="e.g., 1.2, 3.4, 5.6, 2.1, 4.5...")
        
        if st.button("Load Data", type="primary"):
            try:
                # Parse the input - handle both comma and space separation
                data_str = data_input.replace(',', ' ')
                data_values = [float(x) for x in data_str.split() if x.strip()]
                
                if len(data_values) < 2:
                    st.error("Please enter at least 2 data points.")
                else:
                    st.session_state.data = np.array(data_values)
                    st.success(f"✅ Loaded {len(data_values)} data points!")
            except ValueError:
                st.error("Invalid input. Please enter numeric values only.")
    
    else:  # Upload CSV
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview:", df.head())
                
                # Let user select column if multiple columns
                if len(df.columns) > 1:
                    column = st.selectbox("Select data column:", df.columns)
                else:
                    column = df.columns[0]
                
                if st.button("Load Selected Column", type="primary"):
                    data_values = df[column].dropna().values
                    st.session_state.data = data_values
                    st.success(f"✅ Loaded {len(data_values)} data points from '{column}'!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    
    # Generate sample data option
    st.markdown("---")
    st.markdown("**Or generate sample data:**")
    if st.button("Generate Sample Data (Gamma)"):
        st.session_state.data = stats.gamma.rvs(5, 1, 1, size=500)
        st.success("✅ Generated 500 sample points!")

# Main content area
if st.session_state.data is not None:
    data = st.session_state.data
    
    # Display data statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sample Size", len(data))
    with col2:
        st.metric("Mean", f"{np.mean(data):.3f}")
    with col3:
        st.metric("Std Dev", f"{np.std(data):.3f}")
    with col4:
        st.metric("Range", f"{np.min(data):.2f} to {np.max(data):.2f}")
    
    st.markdown("---")
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["🤖 Automatic Fitting", "🎚️ Manual Fitting"])
    
    with tab1:
        st.subheader("Automatic Distribution Fitting")
        
        # Distribution selection
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            selected_dist = st.selectbox(
                "Select Distribution:",
                list(DISTRIBUTIONS.keys()),
                index=1  # Default to Gamma
            )
            
            num_bins = st.slider("Number of histogram bins:", 10, 100, 30)
            
            if st.button("Fit Distribution", type="primary"):
                with st.spinner("Fitting distribution..."):
                    try:
                        dist_obj = DISTRIBUTIONS[selected_dist]
                        
                        # Fit the distribution
                        params = dist_obj.fit(data)
                        
                        # Create the fitted distribution
                        fitted_dist = dist_obj(*params)
                        
                        # Store in session state
                        st.session_state.fitted_params = params
                        st.session_state.fitted_dist = fitted_dist
                        st.session_state.selected_dist = selected_dist
                        st.session_state.dist_obj = dist_obj
                        
                        st.success("✅ Distribution fitted successfully!")
                    except Exception as e:
                        st.error(f"Error fitting distribution: {e}")
        
        with col_right:
            if 'fitted_dist' in st.session_state:
                # Create visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot histogram
                ax.hist(data, bins=num_bins, density=True, alpha=0.6, 
                       color='steelblue', edgecolor='black', label='Data')
                
                # Plot fitted distribution
                x_range = np.linspace(data.min(), data.max(), 200)
                fitted_pdf = st.session_state.fitted_dist.pdf(x_range)
                ax.plot(x_range, fitted_pdf, 'r-', linewidth=2, 
                       label=f'Fitted {st.session_state.selected_dist}')
                
                ax.set_xlabel('Value', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.set_title(f'{st.session_state.selected_dist} Distribution Fit', 
                           fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
                
                # Display parameters and fit quality
                st.markdown("### 📊 Fitting Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Fitted Parameters:**")
                    params = st.session_state.fitted_params
                    param_names = ['shape', 'loc', 'scale'] if len(params) == 3 else [f'param_{i}' for i in range(len(params))]
                    
                    for i, (name, value) in enumerate(zip(param_names, params)):
                        st.write(f"- **{name}:** {value:.6f}")
                
                with col2:
                    st.markdown("**Goodness of Fit:**")
                    
                    # Calculate fit quality metrics
                    # Use histogram comparison
                    hist_data, bin_edges = np.histogram(data, bins=num_bins, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    fitted_values = st.session_state.fitted_dist.pdf(bin_centers)
                    
                    # Calculate errors
                    errors = hist_data - fitted_values
                    mae = np.mean(np.abs(errors))
                    max_error = np.max(np.abs(errors))
                    rmse = np.sqrt(np.mean(errors**2))
                    
                    st.write(f"- **Mean Absolute Error:** {mae:.6f}")
                    st.write(f"- **Root Mean Square Error:** {rmse:.6f}")
                    st.write(f"- **Maximum Error:** {max_error:.6f}")
                    
                    # Kolmogorov-Smirnov test
                    ks_statistic, ks_pvalue = stats.kstest(data, st.session_state.fitted_dist.cdf)
                    st.write(f"- **K-S Statistic:** {ks_statistic:.6f}")
                    st.write(f"- **K-S p-value:** {ks_pvalue:.6f}")
    
    with tab2:
        st.subheader("Manual Distribution Fitting")
        st.markdown("Adjust parameters manually using sliders to explore different fits.")
        
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            manual_dist = st.selectbox(
                "Select Distribution:",
                list(DISTRIBUTIONS.keys()),
                key='manual_dist',
                index=1
            )
            
            dist_obj = DISTRIBUTIONS[manual_dist]
            
            # Get initial parameters from automatic fit if available
            if 'fitted_params' in st.session_state and st.session_state.selected_dist == manual_dist:
                initial_params = st.session_state.fitted_params
            else:
                # Use default fitting
                try:
                    initial_params = dist_obj.fit(data)
                except:
                    initial_params = (1.0, 0.0, 1.0)
            
            st.markdown("**Adjust Parameters:**")
            
            # Create sliders based on number of parameters
            manual_params = []
            
            if len(initial_params) >= 1:
                shape = st.slider("Shape parameter:", 0.1, 20.0, float(initial_params[0]), 0.1)
                manual_params.append(shape)
            
            if len(initial_params) >= 2:
                loc = st.slider("Location (loc):", 
                              float(data.min() - 1), float(data.max() + 1), 
                              float(initial_params[1] if len(initial_params) > 1 else 0), 0.1)
                manual_params.append(loc)
            
            if len(initial_params) >= 3:
                scale = st.slider("Scale parameter:", 0.1, 10.0, 
                                float(initial_params[2] if len(initial_params) > 2 else 1), 0.1)
                manual_params.append(scale)
            
            num_bins_manual = st.slider("Number of histogram bins:", 10, 100, 30, key='bins_manual')
        
        with col_right:
            # Create manual fit visualization
            try:
                manual_fitted_dist = dist_obj(*manual_params)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot histogram
                ax.hist(data, bins=num_bins_manual, density=True, alpha=0.6, 
                       color='steelblue', edgecolor='black', label='Data')
                
                # Plot manually fitted distribution
                x_range = np.linspace(data.min(), data.max(), 200)
                manual_pdf = manual_fitted_dist.pdf(x_range)
                ax.plot(x_range, manual_pdf, 'g-', linewidth=2, 
                       label=f'Manual {manual_dist}')
                
                ax.set_xlabel('Value', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.set_title(f'Manual {manual_dist} Distribution Fit', 
                           fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
                
                # Display manual fit quality
                st.markdown("### 📊 Manual Fit Quality")
                
                hist_data, bin_edges = np.histogram(data, bins=num_bins_manual, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                fitted_values = manual_fitted_dist.pdf(bin_centers)
                
                errors = hist_data - fitted_values
                mae = np.mean(np.abs(errors))
                rmse = np.sqrt(np.mean(errors**2))
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{mae:.6f}")
                col2.metric("RMSE", f"{rmse:.6f}")
                col3.metric("Max Error", f"{np.max(np.abs(errors)):.6f}")
                
            except Exception as e:
                st.error(f"Error creating manual fit: {e}")

else:
    # No data loaded yet
    st.info("👈 Please load data using the sidebar to get started!")
    
    # Show some instructions
    st.markdown("""
    ### Getting Started
    
    1. **Load Your Data** - Use the sidebar to either:
       - Enter data manually (comma or space separated)
       - Upload a CSV file
       - Generate sample data for testing
    
    2. **Automatic Fitting** - Select a distribution and let the app find optimal parameters
    
    3. **Manual Fitting** - Fine-tune parameters using interactive sliders
    
    ### Available Distributions
    
    This tool supports 12 different probability distributions:
    - Normal, Gamma, Weibull, Exponential
    - Log-Normal, Beta, Chi-Square, Uniform
    - Rayleigh, Pareto, Laplace, Cauchy
    
    ### Goodness of Fit Metrics
    
    - **Mean Absolute Error (MAE)** - Average deviation between data and fit
    - **Root Mean Square Error (RMSE)** - Standard deviation of residuals
    - **Maximum Error** - Largest single deviation
    - **Kolmogorov-Smirnov Test** - Statistical test for distribution match
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Data Science Project • 2024")