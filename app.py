import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Marketing Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-positive {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-negative {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-suggestion {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process all CSV files"""
    try:
        # Load datasets
        facebook_df = pd.read_csv('Facebook.csv')
        google_df = pd.read_csv('Google.csv')
        tiktok_df = pd.read_csv('TikTok.csv')
        business_df = pd.read_csv('Business.csv')
        
        # Add platform column to each dataset
        facebook_df['platform'] = 'Facebook'
        google_df['platform'] = 'Google'
        tiktok_df['platform'] = 'TikTok'
        
        # Combine marketing data
        marketing_df = pd.concat([facebook_df, google_df, tiktok_df], ignore_index=True)
        
        # Clean and convert date columns
        marketing_df['date'] = pd.to_datetime(marketing_df['date'])
        business_df['date'] = pd.to_datetime(business_df['date'])
        
        # Rename business columns to match expected format
        business_df = business_df.rename(columns={
            '# of orders': 'orders',
            '# of new orders': 'new_orders',
            'total revenue': 'total_revenue',
            'gross profit': 'gross_profit'
        })
        
        # Calculate derived metrics for marketing data
        marketing_df['CPM'] = np.where(marketing_df['impression'] > 0, 
                                     (marketing_df['spend'] / (marketing_df['impression'] / 1000)), 0)
        marketing_df['CPC'] = np.where(marketing_df['clicks'] > 0, 
                                     marketing_df['spend'] / marketing_df['clicks'], 0)
        marketing_df['ROAS'] = np.where(marketing_df['spend'] > 0, 
                                      marketing_df['attributed revenue'] / marketing_df['spend'], 0)
        marketing_df['CTR'] = np.where(marketing_df['impression'] > 0, 
                                     (marketing_df['clicks'] / marketing_df['impression']) * 100, 0)
        
        # Calculate daily aggregations for marketing
        daily_marketing = marketing_df.groupby(['date', 'platform']).agg({
            'impression': 'sum',
            'clicks': 'sum',
            'spend': 'sum',
            'attributed revenue': 'sum'
        }).reset_index()
        
        # Recalculate metrics for daily data
        daily_marketing['CPM'] = np.where(daily_marketing['impression'] > 0, 
                                        (daily_marketing['spend'] / (daily_marketing['impression'] / 1000)), 0)
        daily_marketing['CPC'] = np.where(daily_marketing['clicks'] > 0, 
                                        daily_marketing['spend'] / daily_marketing['clicks'], 0)
        daily_marketing['ROAS'] = np.where(daily_marketing['spend'] > 0, 
                                         daily_marketing['attributed revenue'] / daily_marketing['spend'], 0)
        daily_marketing['CTR'] = np.where(daily_marketing['impression'] > 0, 
                                        (daily_marketing['clicks'] / daily_marketing['impression']) * 100, 0)
        
        # Merge with business data
        combined_df = daily_marketing.merge(business_df, on='date', how='left')
        
        # Calculate additional business metrics
        combined_df['gross_margin_pct'] = np.where(combined_df['total_revenue'] > 0, 
                                                 (combined_df['gross_profit'] / combined_df['total_revenue']) * 100, 0)
        
        # Calculate CAC (Customer Acquisition Cost) - estimate new customers per platform
        total_daily_marketing = marketing_df.groupby('date').agg({
            'spend': 'sum'
        }).reset_index()
        
        business_with_total_spend = business_df.merge(total_daily_marketing, on='date', how='left')
        business_with_total_spend['CAC'] = np.where(business_with_total_spend['new_customers'] > 0,
                                                  business_with_total_spend['spend'] / business_with_total_spend['new_customers'], 0)
        
        # For platform-specific CAC, distribute new customers proportionally by spend
        platform_spend_ratio = daily_marketing.merge(total_daily_marketing, on='date', suffixes=('', '_total'))
        platform_spend_ratio['spend_ratio'] = np.where(platform_spend_ratio['spend_total'] > 0,
                                                      platform_spend_ratio['spend'] / platform_spend_ratio['spend_total'], 0)
        
        combined_df = combined_df.merge(business_df[['date', 'new_customers']], on='date', how='left')
        combined_df['estimated_new_customers'] = combined_df['spend_ratio'] * combined_df['new_customers']
        combined_df['CAC'] = np.where(combined_df['estimated_new_customers'] > 0,
                                    combined_df['spend'] / combined_df['estimated_new_customers'], 0)
        
        return combined_df, marketing_df, business_df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure all CSV files (Facebook.csv, Google.csv, TikTok.csv, Business.csv) are in the same directory as this app.")
        return None, None, None

def create_kpi_cards(data, date_filter):
    """Create KPI cards"""
    filtered_data = data[(data['date'] >= date_filter[0]) & (data['date'] <= date_filter[1])]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_spend = filtered_data['spend'].sum()
    total_revenue = filtered_data['attributed revenue'].sum()
    avg_roas = total_revenue / total_spend if total_spend > 0 else 0
    avg_cac = filtered_data['CAC'].mean()
    avg_gross_margin = filtered_data['gross_margin_pct'].mean()
    
    with col1:
        st.metric("Total Spend", f"₹{total_spend:,.0f}")
    with col2:
        st.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    with col3:
        st.metric("Average ROAS", f"{avg_roas:.2f}x")
    with col4:
        st.metric("Average CAC", f"₹{avg_cac:.0f}")
    with col5:
        st.metric("Gross Margin %", f"{avg_gross_margin:.1f}%")

def create_trend_charts(data, date_filter):
    """Create time series trend charts"""
    filtered_data = data[(data['date'] >= date_filter[0]) & (data['date'] <= date_filter[1])]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Spend vs Revenue Trends")
        daily_trends = filtered_data.groupby(['date', 'platform']).agg({
            'spend': 'sum',
            'attributed revenue': 'sum'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        for platform in daily_trends['platform'].unique():
            platform_data = daily_trends[daily_trends['platform'] == platform]
            fig.add_trace(
                go.Scatter(x=platform_data['date'], y=platform_data['spend'], 
                          name=f'{platform} Spend', line=dict(dash='dash')),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=platform_data['date'], y=platform_data['attributed revenue'], 
                          name=f'{platform} Revenue'),
                secondary_y=True,
            )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Spend (₹)", secondary_y=False)
        fig.update_yaxes(title_text="Revenue (₹)", secondary_y=True)
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Track how marketing spend translates to attributed revenue across platforms over time.")
    
    with col2:
        st.subheader("👥 New Customers vs Spend")
        
        # Aggregate business data by date for customer trends
        business_trends = filtered_data.groupby('date').agg({
            'spend': 'sum',
            'new_customers': 'first'  # Since it's the same for all platforms on same date
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=business_trends['date'], y=business_trends['spend'], 
                      name='Total Spend', line=dict(color='red', dash='dash')),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=business_trends['date'], y=business_trends['new_customers'], 
                      name='New Customers', line=dict(color='green')),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Spend (₹)", secondary_y=False)
        fig.update_yaxes(title_text="New Customers", secondary_y=True)
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Monitor the relationship between marketing investment and customer acquisition.")

def create_channel_comparison(data, date_filter):
    """Create channel comparison charts"""
    filtered_data = data[(data['date'] >= date_filter[0]) & (data['date'] <= date_filter[1])]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 ROAS by Channel")
        platform_metrics = filtered_data.groupby('platform').agg({
            'spend': 'sum',
            'attributed revenue': 'sum'
        }).reset_index()
        platform_metrics['ROAS'] = platform_metrics['attributed revenue'] / platform_metrics['spend']
        
        fig = px.bar(platform_metrics, x='platform', y='ROAS', 
                     title='Return on Ad Spend by Platform',
                     color='ROAS', color_continuous_scale='RdYlGn')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Higher ROAS indicates better revenue generation per dollar spent.")
    
    with col2:
        st.subheader("💰 CAC by Channel")
        platform_cac = filtered_data.groupby('platform')['CAC'].mean().reset_index()
        
        fig = px.bar(platform_cac, x='platform', y='CAC', 
                     title='Customer Acquisition Cost by Platform',
                     color='CAC', color_continuous_scale='RdYlBu_r')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Lower CAC means more cost-efficient customer acquisition.")

def create_marketing_funnel(data, date_filter):
    """Create marketing funnel visualization"""
    filtered_data = data[(data['date'] >= date_filter[0]) & (data['date'] <= date_filter[1])]
    
    st.subheader("🔄 Marketing Funnel Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        funnel_data = filtered_data.groupby('platform').agg({
            'impression': 'sum',
            'clicks': 'sum',
            'attributed revenue': 'sum'
        }).reset_index()
        
        # Create funnel chart for each platform
        fig = go.Figure()
        
        platforms = funnel_data['platform'].unique()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, platform in enumerate(platforms):
            platform_data = funnel_data[funnel_data['platform'] == platform]
            
            # Normalize data for better visualization
            impressions = platform_data['impression'].iloc[0]
            clicks = platform_data['clicks'].iloc[0]
            revenue = platform_data['attributed revenue'].iloc[0]
            
            # Create funnel stages
            stages = ['Impressions', 'Clicks', 'Revenue (₹)']
            values = [impressions, clicks, revenue * 100]  # Scale revenue for visualization
            
            fig.add_trace(go.Funnel(
                y=stages,
                x=values,
                name=platform,
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(height=500, title="Marketing Funnel by Platform")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Funnel Metrics")
        
        total_impressions = filtered_data['impression'].sum()
        total_clicks = filtered_data['clicks'].sum()
        total_revenue = filtered_data['attributed revenue'].sum()
        
        ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        
        st.metric("Total Impressions", f"{total_impressions:,}")
        st.metric("Total Clicks", f"{total_clicks:,}")
        st.metric("Click-Through Rate", f"{ctr:.2f}%")
        st.metric("Total Attributed Revenue", f"₹{total_revenue:,.0f}")
        
        st.caption("💡 Track the customer journey from impression to revenue generation.")

def create_geographic_analysis(marketing_data, date_filter):
    """Create geographic analysis if state data is available"""
    filtered_data = marketing_data[(marketing_data['date'] >= date_filter[0]) & (marketing_data['date'] <= date_filter[1])]
    
    if 'state' in filtered_data.columns:
        st.subheader("🗺️ Performance by State")
        
        state_performance = filtered_data.groupby('state').agg({
            'spend': 'sum',
            'attributed revenue': 'sum',
            'impression': 'sum',
            'clicks': 'sum'
        }).reset_index()
        
        state_performance['ROAS'] = state_performance['attributed revenue'] / state_performance['spend']
        state_performance['CTR'] = (state_performance['clicks'] / state_performance['impression']) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(state_performance, x='state', y='ROAS', 
                        title='ROAS by State', color='ROAS', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(state_performance, x='spend', y='attributed revenue', 
                           size='impression', color='state', 
                           title='Spend vs Revenue by State')
            st.plotly_chart(fig, use_container_width=True)
        
        st.caption("💡 Identify which geographical regions provide the best ROI for targeted campaigns.")

def generate_insights(data, date_filter):
    """Generate automated insights and recommendations"""
    filtered_data = data[(data['date'] >= date_filter[0]) & (data['date'] <= date_filter[1])]
    
    st.header("🔍 Insights & Recommendations")
    
    # Calculate platform performance
    platform_metrics = filtered_data.groupby('platform').agg({
        'spend': 'sum',
        'attributed revenue': 'sum',
        'impression': 'sum',
        'clicks': 'sum',
        'CAC': 'mean'
    }).reset_index()
    
    platform_metrics['ROAS'] = platform_metrics['attributed revenue'] / platform_metrics['spend']
    platform_metrics['CPC'] = platform_metrics['spend'] / platform_metrics['clicks']
    platform_metrics['CTR'] = (platform_metrics['clicks'] / platform_metrics['impression']) * 100
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📈 Performance Summary")
        
        # Best performing platform
        best_roas_platform = platform_metrics.loc[platform_metrics['ROAS'].idxmax()]
        lowest_cac_platform = platform_metrics.loc[platform_metrics['CAC'].idxmin()]
        
        st.markdown(f"""
        <div class="insight-positive">
            <strong>🟢 Top Performer:</strong><br>
            • Highest ROAS: {best_roas_platform['platform']} ({best_roas_platform['ROAS']:.2f}x)<br>
            • Lowest CAC: {lowest_cac_platform['platform']} (₹{lowest_cac_platform['CAC']:.0f})
        </div>
        """, unsafe_allow_html=True)
        
        # Performance alerts
        poor_roas_platforms = platform_metrics[platform_metrics['ROAS'] < 1.0]
        if not poor_roas_platforms.empty:
            for _, platform in poor_roas_platforms.iterrows():
                st.markdown(f"""
                <div class="insight-negative">
                    <strong>🔴 Alert:</strong><br>
                    {platform['platform']} ROAS is below 1.0x ({platform['ROAS']:.2f}x)<br>
                    Consider budget reallocation or campaign optimization.
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("💡 Recommendations")
        
        # Budget reallocation recommendations
        if len(platform_metrics) > 1:
            best_platform = platform_metrics.loc[platform_metrics['ROAS'].idxmax()]
            worst_platform = platform_metrics.loc[platform_metrics['ROAS'].idxmin()]
            
            if best_platform['ROAS'] > worst_platform['ROAS'] * 1.5:
                st.markdown(f"""
                <div class="insight-suggestion">
                    <strong>💡 Budget Optimization:</strong><br>
                    Consider shifting 10-15% budget from {worst_platform['platform']} 
                    to {best_platform['platform']} for better ROI.
                </div>
                """, unsafe_allow_html=True)
        
        # CPC comparison
        min_cpc_platform = platform_metrics.loc[platform_metrics['CPC'].idxmin()]
        max_cpc_platform = platform_metrics.loc[platform_metrics['CPC'].idxmax()]
        
        if min_cpc_platform['CPC'] < max_cpc_platform['CPC'] * 0.7:
            st.markdown(f"""
            <div class="insight-suggestion">
                <strong>💡 Cost Efficiency:</strong><br>
                {min_cpc_platform['platform']} has 30%+ lower CPC than {max_cpc_platform['platform']}.<br>
                Consider increasing budget allocation to capitalize on lower costs.
            </div>
            """, unsafe_allow_html=True)
        
        # CTR recommendations
        low_ctr_platforms = platform_metrics[platform_metrics['CTR'] < 2.0]
        if not low_ctr_platforms.empty:
            for _, platform in low_ctr_platforms.iterrows():
                st.markdown(f"""
                <div class="insight-suggestion">
                    <strong>💡 Creative Optimization:</strong><br>
                    {platform['platform']} CTR is {platform['CTR']:.2f}% (below 2%).<br>
                    Consider refreshing ad creatives or refining audience targeting.
                </div>
                """, unsafe_allow_html=True)
    
    # Detailed metrics table
    st.subheader("📊 Detailed Platform Metrics")
    display_metrics = platform_metrics.copy()
    display_metrics['spend'] = display_metrics['spend'].apply(lambda x: f"₹{x:,.0f}")
    display_metrics['attributed revenue'] = display_metrics['attributed revenue'].apply(lambda x: f"₹{x:,.0f}")
    display_metrics['ROAS'] = display_metrics['ROAS'].apply(lambda x: f"{x:.2f}x")
    display_metrics['CPC'] = display_metrics['CPC'].apply(lambda x: f"₹{x:.2f}")
    display_metrics['CAC'] = display_metrics['CAC'].apply(lambda x: f"₹{x:.0f}")
    display_metrics['CTR'] = display_metrics['CTR'].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(display_metrics, use_container_width=True)

def main():
    """Main application function"""
    st.title("📊 Marketing Intelligence Dashboard")
    st.markdown("### Transform marketing data into actionable business insights")
    
    # Load data
    combined_data, marketing_data, business_data = load_and_process_data()
    
    if combined_data is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    # Date range filter
    min_date = combined_data['date'].min()
    max_date = combined_data['date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Platform filter
    platforms = ['All'] + list(combined_data['platform'].unique())
    selected_platform = st.sidebar.selectbox("Select Platform", platforms)
    
    # Apply platform filter
    if selected_platform != 'All':
        combined_data = combined_data[combined_data['platform'] == selected_platform]
        marketing_data = marketing_data[marketing_data['platform'] == selected_platform]
    
    # Convert date_range to datetime
    if len(date_range) == 2:
        date_filter = (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
    else:
        date_filter = (min_date, max_date)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "🎯 Channels", "🔍 Insights"])
    
    with tab1:
        st.header("📊 Key Performance Indicators")
        create_kpi_cards(combined_data, date_filter)
        
        st.markdown("---")
        
        # Marketing funnel
        create_marketing_funnel(combined_data, date_filter)
        
        # Geographic analysis if available
        create_geographic_analysis(marketing_data, date_filter)
    
    with tab2:
        st.header("📈 Performance Trends")
        create_trend_charts(combined_data, date_filter)
        
        # Additional trend metrics
        st.subheader("📊 Daily Performance Metrics")
        
        daily_summary = combined_data[(combined_data['date'] >= date_filter[0]) & 
                                    (combined_data['date'] <= date_filter[1])].groupby('date').agg({
            'spend': 'sum',
            'attributed revenue': 'sum',
            'impression': 'sum',
            'clicks': 'sum',
            'ROAS': 'mean',
            'CAC': 'mean'
        }).reset_index()
        
        # Show recent performance
        st.subheader("📅 Recent Daily Performance")
        recent_data = daily_summary.tail(10).copy()
        recent_data['spend'] = recent_data['spend'].apply(lambda x: f"₹{x:,.0f}")
        recent_data['attributed revenue'] = recent_data['attributed revenue'].apply(lambda x: f"₹{x:,.0f}")
        recent_data['ROAS'] = recent_data['ROAS'].apply(lambda x: f"{x:.2f}x")
        recent_data['CAC'] = recent_data['CAC'].apply(lambda x: f"₹{x:.0f}")
        
        st.dataframe(recent_data, use_container_width=True)
    
    with tab3:
        st.header("🎯 Channel Performance Analysis")
        create_channel_comparison(combined_data, date_filter)
        
        # Channel deep dive
        st.subheader("🔍 Channel Deep Dive")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Spend distribution
            platform_spend = combined_data[(combined_data['date'] >= date_filter[0]) & 
                                         (combined_data['date'] <= date_filter[1])].groupby('platform')['spend'].sum()
            
            fig = px.pie(values=platform_spend.values, names=platform_spend.index, 
                        title="Spend Distribution by Platform")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Revenue distribution
            platform_revenue = combined_data[(combined_data['date'] >= date_filter[0]) & 
                                           (combined_data['date'] <= date_filter[1])].groupby('platform')['attributed revenue'].sum()
            
            fig = px.pie(values=platform_revenue.values, names=platform_revenue.index, 
                        title="Revenue Distribution by Platform")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        generate_insights(combined_data, date_filter)
    
    # Footer
    st.markdown("---")
    st.markdown("### 📋 Dashboard Summary")
    
    total_records = len(combined_data[(combined_data['date'] >= date_filter[0]) & 
                                    (combined_data['date'] <= date_filter[1])])
    date_range_str = f"{date_filter[0].strftime('%Y-%m-%d')} to {date_filter[1].strftime('%Y-%m-%d')}"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📅 **Date Range:** {date_range_str}")
    with col2:
        st.info(f"📊 **Records Analyzed:** {total_records:,}")
    with col3:
        st.info(f"🎯 **Platforms:** {selected_platform}")

if __name__ == "__main__":
    main()