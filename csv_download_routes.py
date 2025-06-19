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

def generate_html_report(analytics_data: Dict[str, Any], report_type: str) -> str:
    """Generate HTML report with charts"""
    
    if "error" in analytics_data:
        return f"<html><body><h1>Error generating {report_type} report</h1><p>{analytics_data['error']}</p></body></html>"
    
    # Prepare chart data
    product_labels = list(analytics_data['product_sales'].keys())
    product_values = [analytics_data['product_sales'][p]['total_value'] for p in product_labels]
    
    monthly_labels = list(analytics_data['monthly_sales'].keys())
    monthly_values = [analytics_data['monthly_sales'][m]['total_value'] for m in monthly_labels]
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sales Analytics Report - {datetime.now().strftime('%Y-%m-%d')}</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
            .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .summary-card {{ background-color: #e9f5ff; padding: 15px; border-radius: 5px; text-align: center; }}
            .chart-container {{ width: 48%; display: inline-block; margin: 1%; }}
            .table-container {{ margin: 20px 0; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .export-btn {{ background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Sales Analytics Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Report Type: {report_type.title()}</p>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <h3>💰 Total Revenue</h3>
                <h2>${analytics_data['summary']['total_revenue']:,.2f}</h2>
            </div>
            <div class="summary-card">
                <h3>📈 Total Transactions</h3>
                <h2>{analytics_data['summary']['total_transactions']:,}</h2>
            </div>
            <div class="summary-card">
                <h3>💵 Avg Transaction</h3>
                <h2>${analytics_data['summary']['avg_transaction_value']:,.2f}</h2>
            </div>
            <div class="summary-card">
                <h3>🏆 Best Seller</h3>
                <h2>{analytics_data['summary']['best_selling_product']}</h2>
            </div>
            <div class="summary-card">
                <h3>💎 Top Revenue Product</h3>
                <h2>{analytics_data['summary']['highest_revenue_product']}</h2>
            </div>
            <div class="summary-card">
                <h3>📅 Last 30 Days</h3>
                <h2>${analytics_data['summary']['recent_30_days_revenue']:,.2f}</h2>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Sales by Product (Revenue)</h3>
            <canvas id="productPieChart" width="400" height="400"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>Monthly Sales Trend</h3>
            <canvas id="monthlyLineChart" width="400" height="400"></canvas>
        </div>
        
        <div class="table-container">
            <h3>Top 10 Products by Revenue</h3>
            <table>
                <thead>
                    <tr>
                        <th>Product</th>
                        <th>Total Revenue</th>
                        <th>Total Quantity Sold</th>
                        <th>Avg Price per Unit</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add top products table
    for product, data in list(analytics_data['top_products'].items())[:10]:
        avg_price = data['total_value'] / data['quantity'] if data['quantity'] > 0 else 0
        html_content += f"""
                    <tr>
                        <td>{product}</td>
                        <td>${data['total_value']:,.2f}</td>
                        <td>{data['quantity']:,}</td>
                        <td>${avg_price:.2f}</td>
                    </tr>
        """
    
    html_content += f"""
                </tbody>
            </table>
        </div>
        
        <div class="table-container">
            <h3>Recent Sales Transactions</h3>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Product</th>
                        <th>Quantity</th>
                        <th>Unit Price</th>
                        <th>Total Value</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add recent transactions (last 20)
    for transaction in analytics_data['raw_data'][:20]:
        date_str = transaction['date'].strftime('%Y-%m-%d') if isinstance(transaction['date'], datetime) else str(transaction['date'])
        html_content += f"""
                    <tr>
                        <td>{date_str}</td>
                        <td>{transaction['product']}</td>
                        <td>{transaction['quantity']}</td>
                        <td>${transaction['unit_price']:.2f}</td>
                        <td>${transaction['total_value']:.2f}</td>
                    </tr>
        """
    
    html_content += f"""
                </tbody>
            </table>
        </div>
        
        <script>
            // Product Pie Chart
            const productCtx = document.getElementById('productPieChart').getContext('2d');
            new Chart(productCtx, {{
                type: 'pie',
                data: {{
                    labels: {json.dumps(product_labels[:10])},
                    datasets: [{{
                        data: {json.dumps(product_values[:10])},
                        backgroundColor: [
                            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
                            '#FF9F40', '#FF6384', '#C9CBCF', '#4BC0C0', '#FF6384'
                        ]
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            position: 'bottom'
                        }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    return context.label + ': $' + context.parsed.toFixed(2);
                                }}
                            }}
                        }}
                    }}
                }}
            }});
            
            // Monthly Line Chart
            const monthlyCtx = document.getElementById('monthlyLineChart').getContext('2d');
            new Chart(monthlyCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(monthly_labels)},
                    datasets: [{{
                        label: 'Monthly Revenue',
                        data: {json.dumps(monthly_values)},
                        borderColor: '#36A2EB',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        fill: true,
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            display: true
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            ticks: {{
                                callback: function(value) {{
                                    return '$' + value.toFixed(2);
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        </script>
        
        <div style="margin-top: 30px; text-align: center;">
            <p><em>This report was automatically generated by your Sales Analytics System</em></p>
        </div>
    </body>
    </html>
    """
    
    return html_content

@router.get("/download-enhanced-report/{report_type}")
async def download_enhanced_report(report_type: str, current_user: dict = Depends(get_current_user)):
    """Download enhanced HTML report with visualizations"""
    try:
        if report_type == "sales":
            analytics_data = await generate_sales_analytics(current_user)
            html_content = generate_html_report(analytics_data, report_type)
            filename = f"sales_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
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