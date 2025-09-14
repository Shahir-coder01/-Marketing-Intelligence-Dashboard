# 📊 Marketing Intelligence Dashboard  

An interactive BI dashboard that transforms **marketing campaign data** (Facebook, Google, TikTok) and **business performance data** into actionable insights.  

Built with **Streamlit** and **Plotly**, this app helps decision-makers understand how marketing activity connects with revenue, customers, and profit.  

---

## 🚀 Features  

- 📊 **Key KPIs**: Total Spend, Revenue, ROAS, CAC, Gross Margin %  
- 📈 **Trends**: Marketing spend vs revenue, new customers vs spend  
- 🎯 **Channel Analysis**: ROAS & CAC by platform (Facebook, Google, TikTok)  
- 🔄 **Marketing Funnel**: Impressions → Clicks → Orders → Revenue  
- 🗺️ **Geographic Analysis**: ROI by state  
- 🔍 **Insights & Recommendations**: Suggestions to optimize spend and boost ROI  

---

## 📂 Project Structure  

├── app.py # Streamlit app code
├── requirements.txt # Python dependencies
├── # Input CSV files (Facebook, Google, TikTok, Business)
└── README.md # Project documentation


## 🛠️ Installation & Setup  

### 1. Clone the Repository  

git clone https://github.com/your-username/marketing-intelligence-dashboard.git
cd marketing-intelligence-dashboard

### 2. (Optional) Create Virtual Environment
   
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Run the Dashboard
   
streamlit run app.py

Then open the provided localhost URL in your browser (default: http://localhost:8501).

