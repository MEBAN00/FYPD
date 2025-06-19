from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from auth import get_current_user, db
import csv
import pandas as pd
import json
import io
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any

router = APIRouter()

async def generate_sales_analytics(current_user: dict) -> Dict[str, Any]:
    """Generate comprehensive sales analytics"""
    try:
        sales_collection = db["FYPDS"]
        
        # Get all sales data
        sales = await sales_collection.find().sort("date", -1).to_list(length=1000)
        
        if not sales:
            return {"error": "No sales data available"}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(sales)
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        df['total_value'] = pd.to_numeric(df['total_value'])
        df['quantity'] = pd.to_numeric(df['quantity'])
        
        # Sales by product (for pie chart)
        product_sales = df.groupby('product').agg({
            'total_value': 'sum',
            'quantity': 'sum'
        }).round(2)
        
        # Monthly sales trend
        monthly_sales = df.groupby('month').agg({
            'total_value': 'sum',
            'quantity': 'sum'
        }).round(2)
        
        # Top performing products
        top_products = product_sales.sort_values('total_value', ascending=False).head(10)
        
        # Sales summary statistics
        total_revenue = df['total_value'].sum()
        total_transactions = len(df)
        avg_transaction_value = df['total_value'].mean()
        best_selling_product = product_sales['quantity'].idxmax()
        highest_revenue_product = product_sales['total_value'].idxmax()
        
        # Recent sales trend (last 30 days)
        recent_date = datetime.now() - timedelta(days=30)
        recent_sales = df[df['date'] >= recent_date]
        recent_revenue = recent_sales['total_value'].sum() if not recent_sales.empty else 0
        
        return {
            "summary": {
                "total_revenue": round(float(total_revenue), 2),
                "total_transactions": total_transactions,
                "avg_transaction_value": round(float(avg_transaction_value), 2),
                "best_selling_product": best_selling_product,
                "highest_revenue_product": highest_revenue_product,
                "recent_30_days_revenue": round(float(recent_revenue), 2)
            },
            "product_sales": product_sales.to_dict('index'),
            "monthly_sales": {str(k): v for k, v in monthly_sales.to_dict('index').items()},
            "top_products": top_products.to_dict('index'),
            "raw_data": df.to_dict('records')
        }
        
    except Exception as e:
        logging.error(f"Error generating sales analytics: {str(e)}")
        return {"error": str(e)}

def generate_futuristic_html_report(analytics_data: Dict[str, Any], report_type: str) -> str:
    """Generate modern, futuristic HTML report with charts"""
    
    if "error" in analytics_data:
        return f"<html><body><h1>Error generating {report_type} report</h1><p>{analytics_data['error']}</p></body></html>"
    
    # Prepare chart data
    product_labels = list(analytics_data['product_sales'].keys())
    product_values = [analytics_data['product_sales'][p]['total_value'] for p in product_labels]
    
    monthly_labels = list(analytics_data['monthly_sales'].keys())
    monthly_values = [analytics_data['monthly_sales'][m]['total_value'] for m in monthly_labels]
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sales Analytics Report - {datetime.now().strftime('%Y-%m-%d')}</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #ffffff;
                overflow-x: hidden;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 2rem;
            }}
            
            .header {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 24px;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                position: relative;
                overflow: hidden;
            }}
            
            .header::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, #00f5ff, #ff00ff, #00ff00, #ffff00);
                animation: shimmer 3s linear infinite;
            }}
            
            @keyframes shimmer {{
                0% {{ transform: translateX(-100%); }}
                100% {{ transform: translateX(100%); }}
            }}
            
            .header h1 {{
                font-size: 3rem;
                font-weight: 800;
                background: linear-gradient(135deg, #00f5ff, #ff00ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                gap: 1rem;
            }}
            
            .header-icon {{
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, #00f5ff, #ff00ff);
                border-radius: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.5rem;
                color: white;
                box-shadow: 0 8px 24px rgba(0, 245, 255, 0.3);
            }}
            
            .header-meta {{
                display: flex;
                gap: 2rem;
                font-size: 1rem;
                opacity: 0.8;
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 1.5rem;
                margin-bottom: 2rem;
            }}
            
            .stat-card {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 20px;
                padding: 1.5rem;
                position: relative;
                overflow: hidden;
                transition: all 0.3s ease;
                cursor: pointer;
            }}
            
            .stat-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
                border-color: rgba(255, 255, 255, 0.4);
            }}
            
            .stat-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 1px;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
            }}
            
            .stat-icon {{
                width: 48px;
                height: 48px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.2rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            }}
            
            .stat-icon.revenue {{ background: linear-gradient(135deg, #4ade80, #22c55e); }}
            .stat-icon.transactions {{ background: linear-gradient(135deg, #3b82f6, #1d4ed8); }}
            .stat-icon.average {{ background: linear-gradient(135deg, #f59e0b, #d97706); }}
            .stat-icon.product {{ background: linear-gradient(135deg, #8b5cf6, #7c3aed); }}
            .stat-icon.top {{ background: linear-gradient(135deg, #ef4444, #dc2626); }}
            .stat-icon.recent {{ background: linear-gradient(135deg, #06b6d4, #0891b2); }}
            
            .stat-label {{
                font-size: 0.875rem;
                opacity: 0.8;
                margin-bottom: 0.5rem;
                font-weight: 500;
            }}
            
            .stat-value {{
                font-size: 1.75rem;
                font-weight: 700;
                line-height: 1.2;
            }}
            
            .charts-section {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 2rem;
                margin-bottom: 2rem;
            }}
            
            .chart-card {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 20px;
                padding: 1.5rem;
                position: relative;
                overflow: hidden;
            }}
            
            .chart-card h3 {{
                font-size: 1.25rem;
                font-weight: 600;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            
            .chart-container {{
                height: 300px;
                position: relative;
            }}
            
            .table-section {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 20px;
                padding: 1.5rem;
                margin-bottom: 2rem;
                overflow: hidden;
            }}
            
            .table-section h3 {{
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            
            .modern-table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                overflow: hidden;
                margin-bottom: 2rem;
            }}
            
            .modern-table th {{
                background: rgba(255, 255, 255, 0.1);
                padding: 1rem;
                text-align: left;
                font-weight: 600;
                font-size: 0.875rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .modern-table td {{
                padding: 1rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                transition: background-color 0.2s ease;
            }}
            
            .modern-table tr:hover td {{
                background: rgba(255, 255, 255, 0.05);
            }}
            
            .modern-table tr:last-child td {{
                border-bottom: none;
            }}
            
            .footer {{
                text-align: center;
                padding: 2rem;
                opacity: 0.6;
                font-size: 0.875rem;
            }}
            
            .glow {{
                box-shadow: 0 0 20px rgba(0, 245, 255, 0.3);
            }}
            
            @media (max-width: 768px) {{
                .charts-section {{
                    grid-template-columns: 1fr;
                }}
                
                .stats-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .header h1 {{
                    font-size: 2rem;
                }}
                
                .container {{
                    padding: 1rem;
                }}
            }}
            
            .pulse {{
                animation: pulse 2s infinite;
            }}
            
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.7; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <div class="header">
                <h1>
                    <div class="header-icon">
                        <i class="fas fa-chart-line"></i>
                    </div>
                    Sales Analytics Report
                </h1>
                <div class="header-meta">
                    <div><i class="fas fa-calendar"></i> Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}</div>
                    <div><i class="fas fa-tag"></i> Report Type: {report_type.title()}</div>
                    <div><i class="fas fa-database"></i> Data Points: {len(analytics_data['raw_data'])}</div>
                </div>
            </div>
            
            <!-- Stats Grid -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-icon revenue">
                        <i class="fas fa-dollar-sign"></i>
                    </div>
                    <div class="stat-label">Total Revenue</div>
                    <div class="stat-value">${analytics_data['summary']['total_revenue']:,.2f}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-icon transactions">
                        <i class="fas fa-receipt"></i>
                    </div>
                    <div class="stat-label">Total Transactions</div>
                    <div class="stat-value">{analytics_data['summary']['total_transactions']:,}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-icon average">
                        <i class="fas fa-chart-bar"></i>
                    </div>
                    <div class="stat-label">Average Transaction</div>
                    <div class="stat-value">${analytics_data['summary']['avg_transaction_value']:,.2f}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-icon product">
                        <i class="fas fa-star"></i>
                    </div>
                    <div class="stat-label">Best Selling Product</div>
                    <div class="stat-value" style="font-size: 1.2rem;">{analytics_data['summary']['best_selling_product']}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-icon top">
                        <i class="fas fa-trophy"></i>
                    </div>
                    <div class="stat-label">Top Revenue Product</div>
                    <div class="stat-value" style="font-size: 1.2rem;">{analytics_data['summary']['highest_revenue_product']}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-icon recent">
                        <i class="fas fa-clock"></i>
                    </div>
                    <div class="stat-label">Last 30 Days Revenue</div>
                    <div class="stat-value">${analytics_data['summary']['recent_30_days_revenue']:,.2f}</div>
                </div>
            </div>
            
            <!-- Charts Section -->
            <div class="charts-section">
                <div class="chart-card">
                    <h3><i class="fas fa-chart-pie"></i> Revenue Distribution</h3>
                    <div class="chart-container">
                        <canvas id="productPieChart"></canvas>
                    </div>
                </div>
                
                <div class="chart-card">
                    <h3><i class="fas fa-chart-line"></i> Monthly Trends</h3>
                    <div class="chart-container">
                        <canvas id="monthlyLineChart"></canvas>
                    </div>
                </div>
            </div>
            
            <!-- Top Products Table -->
            <div class="table-section">
                <h3><i class="fas fa-medal"></i> Top Performing Products</h3>
                <table class="modern-table">
                    <thead>
                        <tr>
                            <th><i class="fas fa-hashtag"></i> Rank</th>
                            <th><i class="fas fa-box"></i> Product</th>
                            <th><i class="fas fa-dollar-sign"></i> Total Revenue</th>
                            <th><i class="fas fa-cubes"></i> Quantity Sold</th>
                            <th><i class="fas fa-calculator"></i> Avg Price</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Add top products table
    for i, (product, data) in enumerate(list(analytics_data['top_products'].items())[:10], 1):
        avg_price = data['total_value'] / data['quantity'] if data['quantity'] > 0 else 0
        rank_icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
        html_content += f"""
                        <tr>
                            <td style="font-weight: 600;">{rank_icon}</td>
                            <td style="font-weight: 500;">{product}</td>
                            <td style="color: #4ade80; font-weight: 600;">${data['total_value']:,.2f}</td>
                            <td>{data['quantity']:,}</td>
                            <td>${avg_price:.2f}</td>
                        </tr>
        """
    
    html_content += f"""
                    </tbody>
                </table>
            </div>
            
            <!-- Recent Transactions Table -->
            <div class="table-section">
                <h3><i class="fas fa-history"></i> Recent Sales Transactions</h3>
                <table class="modern-table">
                    <thead>
                        <tr>
                            <th><i class="fas fa-calendar"></i> Date</th>
                            <th><i class="fas fa-box"></i> Product</th>
                            <th><i class="fas fa-sort-numeric-up"></i> Quantity</th>
                            <th><i class="fas fa-tag"></i> Unit Price</th>
                            <th><i class="fas fa-dollar-sign"></i> Total Value</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Add recent transactions (last 15)
    for transaction in analytics_data['raw_data'][:15]:
        date_str = transaction['date'].strftime('%Y-%m-%d') if isinstance(transaction['date'], datetime) else str(transaction['date'])
        html_content += f"""
                        <tr>
                            <td>{date_str}</td>
                            <td style="font-weight: 500;">{transaction['product']}</td>
                            <td>{transaction['quantity']}</td>
                            <td>${transaction['unit_price']:.2f}</td>
                            <td style="color: #4ade80; font-weight: 600;">${transaction['total_value']:.2f}</td>
                        </tr>
        """
    
    html_content += f"""
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                <p><i class="fas fa-robot"></i> This report was automatically generated by your AI-powered Sales Analytics System</p>
                <p style="margin-top: 0.5rem; opacity: 0.5;">Powered by advanced machine learning algorithms and real-time data processing</p>
            </div>
        </div>
        
        <script>
            // Enhanced Chart.js configuration with futuristic styling
            Chart.defaults.color = '#ffffff';
            Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
            Chart.defaults.backgroundColor = 'rgba(255, 255, 255, 0.1)';
            
            // Product Pie Chart with modern styling
            const productCtx = document.getElementById('productPieChart').getContext('2d');
            new Chart(productCtx, {{
                type: 'doughnut',
                data: {{
                    labels: {json.dumps(product_labels[:8])},
                    datasets: [{{
                        data: {json.dumps(product_values[:8])},
                        backgroundColor: [
                            'rgba(0, 245, 255, 0.8)',
                            'rgba(255, 0, 255, 0.8)', 
                            'rgba(0, 255, 0, 0.8)',
                            'rgba(255, 255, 0, 0.8)',
                            'rgba(255, 100, 100, 0.8)',
                            'rgba(100, 255, 255, 0.8)',
                            'rgba(255, 150, 0, 0.8)',
                            'rgba(150, 100, 255, 0.8)'
                        ],
                        borderColor: [
                            'rgba(0, 245, 255, 1)',
                            'rgba(255, 0, 255, 1)',
                            'rgba(0, 255, 0, 1)',
                            'rgba(255, 255, 0, 1)',
                            'rgba(255, 100, 100, 1)',
                            'rgba(100, 255, 255, 1)',
                            'rgba(255, 150, 0, 1)',
                            'rgba(150, 100, 255, 1)'
                        ],
                        borderWidth: 2,
                        hoverBorderWidth: 4,
                        cutout: '60%'
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'bottom',
                            labels: {{
                                padding: 20,
                                usePointStyle: true,
                                pointStyle: 'circle',
                                font: {{
                                    size: 11,
                                    weight: '500'
                                }}
                            }}
                        }},
                        tooltip: {{
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: '#ffffff',
                            bodyColor: '#ffffff',
                            borderColor: 'rgba(255, 255, 255, 0.2)',
                            borderWidth: 1,
                            cornerRadius: 8,
                            callbacks: {{
                                label: function(context) {{
                                    const percentage = ((context.parsed / {sum(product_values[:8])}) * 100).toFixed(1);
                                    return context.label + ': $' + context.parsed.toFixed(2) + ' (' + percentage + '%)';
                                }}
                            }}
                        }}
                    }},
                    animation: {{
                        animateRotate: true,
                        duration: 2000
                    }}
                }}
            }});
            
            // Monthly Line Chart with futuristic styling
            const monthlyCtx = document.getElementById('monthlyLineChart').getContext('2d');
            new Chart(monthlyCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(monthly_labels)},
                    datasets: [{{
                        label: 'Monthly Revenue',
                        data: {json.dumps(monthly_values)},
                        borderColor: 'rgba(0, 245, 255, 1)',
                        backgroundColor: 'rgba(0, 245, 255, 0.1)',
                        fill: true,
                        tension: 0.4,
                        borderWidth: 3,
                        pointBackgroundColor: 'rgba(0, 245, 255, 1)',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        pointHoverBackgroundColor: 'rgba(255, 0, 255, 1)',
                        pointHoverBorderColor: '#ffffff',
                        pointHoverBorderWidth: 3
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            display: false
                        }},
                        tooltip: {{
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: '#ffffff',
                            bodyColor: '#ffffff',
                            borderColor: 'rgba(255, 255, 255, 0.2)',
                            borderWidth: 1,
                            cornerRadius: 8,
                            callbacks: {{
                                label: function(context) {{
                                    return 'Revenue: $' + context.parsed.y.toFixed(2);
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            grid: {{
                                color: 'rgba(255, 255, 255, 0.1)',
                                drawBorder: false
                            }},
                            ticks: {{
                                color: 'rgba(255, 255, 255, 0.8)',
                                font: {{
                                    size: 11
                                }}
                            }}
                        }},
                        y: {{
                            beginAtZero: true,
                            grid: {{
                                color: 'rgba(255, 255, 255, 0.1)',
                                drawBorder: false
                            }},
                            ticks: {{
                                color: 'rgba(255, 255, 255, 0.8)',
                                font: {{
                                    size: 11
                                }},
                                callback: function(value) {{
                                    return '$' + value.toFixed(0);
                                }}
                            }}
                        }}
                    }},
                    animation: {{
                        duration: 2000,
                        easing: 'easeInOutQuart'
                    }}
                }}
            }});
            
            // Add hover effects to stat cards
            document.querySelectorAll('.stat-card').forEach(card => {{
                card.addEventListener('mouseenter', function() {{
                    this.style.transform = 'translateY(-8px) scale(1.02)';
                    this.style.boxShadow = '0 25px 50px rgba(0, 0, 0, 0.3)';
                }});
                
                card.addEventListener('mouseleave', function() {{
                    this.style.transform = 'translateY(0) scale(1)';
                    this.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.1)';
                }});
            }});
            
            // Add loading animation
            window.addEventListener('load', function() {{
                document.querySelectorAll('.stat-card, .chart-card, .table-section').forEach((element, index) => {{
                    element.style.opacity = '0';
                    element.style.transform = 'translateY(20px)';
                    
                    setTimeout(() => {{
                        element.style.transition = 'all 0.6s ease';
                        element.style.opacity = '1';
                        element.style.transform = 'translateY(0)';
                    }}, index * 100);
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content

@router.get("/download-enhanced-report/{report_type}")
async def download_enhanced_report(report_type: str, current_user: dict = Depends(get_current_user)):
    """Download enhanced futuristic HTML report with visualizations"""
    try:
        if report_type == "sales":
            analytics_data = await generate_sales_analytics(current_user)
            html_content = generate_futuristic_html_report(analytics_data, report_type)
            filename = f"futuristic_sales_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        else:
            raise HTTPException(status_code=400, detail="Invalid report type")
        
        # Create streaming response
        return StreamingResponse(
            io.BytesIO(html_content.encode('utf-8')),
            media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logging.error(f"Error downloading enhanced report: {str(e)}")
        raise HTTPException(status_code=500, detail="Error downloading enhanced report")

async def generate_predicted_profits_csv(current_user: dict) -> str:
    """Generate CSV data for predicted gross profits"""
    try:
        predictions_collection = db["FYPD_PREDICTIONS"]
        
        # Get all predictions
        predictions = await predictions_collection.find().sort("prediction_date", -1).to_list(length=1000)
        
        if not predictions:
            # Create sample predictions data if none exist
            predictions = [
                {
                    "product": "Sample Product 1",
                    "quantity": 100,
                    "unit_price": 25.50,
                    "total_cost": 1500.00,
                    "total_price": 2550.00,
                    "predicted_profit": 1050.00,
                    "prediction_date": datetime.now() - timedelta(days=1),
                    "year": 2024,
                    "user": "sample_user"
                }
            ]
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Product', 'Quantity', 'Unit Price', 'Total Cost', 
            'Total Price', 'Predicted Profit', 'Prediction Date', 'Year', 'User'
        ])
        
        # Write data rows
        for prediction in predictions:
            writer.writerow([
                prediction.get('product', ''),
                prediction.get('quantity', 0),
                prediction.get('unit_price', 0),
                prediction.get('total_cost', 0),
                prediction.get('total_price', 0),
                prediction.get('predicted_profit', 0),
                prediction.get('prediction_date', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                prediction.get('year', 2024),
                prediction.get('user', '')
            ])
        
        return output.getvalue()
        
    except Exception as e:
        logging.error(f"Error generating predicted profits CSV: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating predicted profits data")

async def generate_sales_csv(current_user: dict) -> str:
    """Generate CSV data for sales"""
    try:
        sales_collection = db["FYPDS"]
        
        # Get all sales data
        sales = await sales_collection.find().sort("date", -1).to_list(length=1000)
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Sale ID', 'Product', 'Quantity', 'Unit Price', 
            'Total Value', 'Sale Date'
        ])
        
        # Write data rows
        for sale in sales:
            writer.writerow([
                sale.get('sale_id', ''),
                sale.get('product', ''),
                sale.get('quantity', 0),
                float(sale.get('unit_price', 0)),
                float(sale.get('total_value', 0)),
                sale.get('date', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        return output.getvalue()
        
    except Exception as e:
        logging.error(f"Error generating sales CSV: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating sales data")

async def generate_inventory_updates_csv(current_user: dict) -> str:
    """Generate CSV data for inventory updates only"""
    try:
        # Get inventory updates from the dedicated collection
        inventory_updates_collection = db["FYPD_INVENTORY_UPDATES"]
        
        # Get all inventory updates, sorted by date in descending order
        updates = await inventory_updates_collection.find().sort("update_date", -1).to_list(length=1000)
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header with enhanced fields
        writer.writerow([
            'Product', 'Update Type', 'Quantity Changed', 'Previous Stock', 
            'New Stock', 'Update Date', 'Reason', 'User', 'Additional Details'
        ])
        
        # Write data rows
        for update in updates:
            # Format additional details as a readable string
            additional_details = update.get('additional_details', {})
            details_str = '; '.join([f"{k}: {v}" for k, v in additional_details.items()]) if additional_details else ''
            
            writer.writerow([
                update.get('product', ''),
                update.get('update_type', ''),
                update.get('quantity_changed', 0),
                update.get('previous_stock', 0),
                update.get('new_stock', 0),
                update.get('update_date', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                update.get('reason', ''),
                update.get('user', ''),
                details_str
            ])
        
        return output.getvalue()
        
    except Exception as e:
        logging.error(f"Error generating inventory updates CSV: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating inventory updates data")

@router.get("/download-csv/{csv_type}")
async def download_csv(csv_type: str, current_user: dict = Depends(get_current_user)):
    """Download individual CSV files"""
    try:
        if csv_type == "profits":
            csv_content = await generate_predicted_profits_csv(current_user)
            filename = f"predicted_profits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        elif csv_type == "sales":
            csv_content = await generate_sales_csv(current_user)
            filename = f"sales_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        elif csv_type == "inventory":
            csv_content = await generate_inventory_updates_csv(current_user)
            filename = f"inventory_updates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            raise HTTPException(status_code=400, detail="Invalid CSV type")
        
        # Create streaming response
        return StreamingResponse(
            io.BytesIO(csv_content.encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logging.error(f"Error downloading CSV: {str(e)}")
        raise HTTPException(status_code=500, detail="Error downloading CSV file")

@router.get("/generate-all-csvs")
async def generate_all_csvs(current_user: dict = Depends(get_current_user)):
    """Generate all three CSV files and return their data"""
    try:
        profits_csv = await generate_predicted_profits_csv(current_user)
        sales_csv = await generate_sales_csv(current_user)
        inventory_csv = await generate_inventory_updates_csv(current_user)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return {
            "profits": {
                "filename": f"predicted_profits_{timestamp}.csv",
                "content": profits_csv
            },
            "sales": {
                "filename": f"sales_data_{timestamp}.csv", 
                "content": sales_csv
            },
            "inventory": {
                "filename": f"inventory_updates_{timestamp}.csv",
                "content": inventory_csv
            }
        }
        
    except Exception as e:
        logging.error(f"Error generating all CSVs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating CSV files")