import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_pharmacy_sales(num_records=500000, output_path='omo_sales_data.xlsx'):
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(minutes=int(x)) for x in np.random.uniform(0, 525600, num_records)]
    dates.sort()

    # Product dictionary with base products and prices
    products = {
        'Prescription': [
            ('Lisinopril 10mg', 6.99, 12.99, 20.99, (0.40, 0.50)),
            ('Amlodipine 5mg', 6.49, 11.99, 19.99, (0.35, 0.45)),
            ('Metoprolol 25mg', 7.99, 14.99, 22.99, (0.38, 0.48)),
            ('Atorvastatin 20mg', 12.99, 24.99, 35.99, (0.40, 0.50)),
            ('Losartan 50mg', 8.99, 16.99, 25.99, (0.35, 0.45)),
            ('Valsartan 80mg', 10.99, 19.99, 29.99, (0.38, 0.48)),
            ('Warfarin 5mg', 7.49, 13.99, 21.99, (0.40, 0.50)),

            # Diabetes Medications
            ('Metformin 500mg', 5.99, 10.99, 18.99, (0.35, 0.45)),
            ('Glipizide 5mg', 7.99, 14.99, 23.99, (0.38, 0.48)),
            ('Januvia 100mg', 15.99, 29.99, 45.99, (0.42, 0.52)),
            ('Jardiance 10mg', 16.99, 31.99, 47.99, (0.40, 0.50)),
            
            # Antibiotics
            ('Amoxicillin 500mg', 8.99, 15.99, 25.99, (0.35, 0.45)),
            ('Azithromycin 250mg', 9.99, 18.99, 28.99, (0.38, 0.48)),
            ('Ciprofloxacin 500mg', 8.49, 15.99, 24.99, (0.40, 0.50)),
            ('Doxycycline 100mg', 7.99, 14.99, 23.99, (0.38, 0.48)),

            # Mental Health Medications
            ('Sertraline 50mg', 9.99, 18.99, 28.99, (0.40, 0.50)),
            ('Escitalopram 10mg', 10.99, 20.99, 31.99, (0.38, 0.48)),
            ('Fluoxetine 20mg', 8.99, 16.99, 26.99, (0.40, 0.50)),
            ('Bupropion 150mg', 11.99, 22.99, 33.99, (0.38, 0.48)),
            ('Alprazolam 0.5mg', 7.99, 14.99, 23.99, (0.42, 0.52)),
            
            # Respiratory Medications
            ('Albuterol Inhaler', 15.99, 29.99, 45.99, (0.40, 0.50)),
            ('Fluticasone Nasal', 13.99, 25.99, 39.99, (0.38, 0.48)),
            ('Montelukast 10mg', 12.99, 23.99, 35.99, (0.40, 0.50)),

            # Gastrointestinal Medications
            ('Omeprazole 20mg', 8.99, 16.99, 24.99, (0.35, 0.45)),
            ('Pantoprazole 40mg', 9.99, 18.99, 28.99, (0.38, 0.48)),
            ('Ranitidine 150mg', 7.99, 14.99, 22.99, (0.40, 0.50)),
            
            # Pain Management
            ('Tramadol 50mg', 8.99, 16.99, 25.99, (0.42, 0.52)),
            ('Gabapentin 300mg', 7.49, 13.99, 21.99, (0.38, 0.48)),
            ('Cyclobenzaprine 10mg', 6.99, 12.99, 20.99, (0.40, 0.50)),
            
            # Hormone Medications
            ('Levothyroxine 50mcg', 7.99, 14.99, 22.99, (0.40, 0.48)),
            ('Prednisone 5mg', 6.99, 12.99, 20.99, (0.42, 0.52)),
            ('Estradiol 1mg', 9.99, 18.99, 28.99, (0.40, 0.50))
        ],
        'OTC Medication': [
            # Pain Relief
            ('Acetaminophen 500mg', 3.99, 7.99, 12.99, (0.45, 0.55)),
            ('Ibuprofen 200mg', 3.49, 6.99, 11.99, (0.45, 0.55)),
            ('Naproxen 220mg', 4.49, 8.99, 14.99, (0.48, 0.58)),
            ('Aspirin 325mg', 3.29, 6.49, 10.99, (0.45, 0.55)),
            
            # Cold and Allergy
            ('Cold & Flu Relief', 4.99, 9.99, 16.99, (0.45, 0.55)),
            ('Allergy Relief', 4.49, 8.99, 15.99, (0.48, 0.58)),
            ('Loratadine 10mg', 5.99, 11.99, 18.99, (0.45, 0.55)),
            ('Cetirizine 10mg', 5.49, 10.99, 17.99, (0.48, 0.58)),
            ('Nasal Decongestant', 4.99, 9.99, 16.99, (0.45, 0.55)),
            ('Cough Suppressant', 5.99, 11.99, 18.99, (0.48, 0.58)),
            ('Throat Lozenges', 2.99, 5.99, 9.99, (0.50, 0.60)),
            ('Sinus Relief', 5.49, 10.99, 17.99, (0.48, 0.58)),
            
            # Digestive Health
            ('Antacid Tablets', 2.99, 5.99, 9.99, (0.50, 0.60)),
            ('Acid Reducer', 4.99, 9.99, 16.99, (0.48, 0.58)),
            ('Anti-Diarrheal', 4.49, 8.99, 14.99, (0.45, 0.55)),
            ('Fiber Supplement', 5.99, 11.99, 18.99, (0.48, 0.58)),
            ('Probiotics', 8.99, 17.99, 27.99, (0.45, 0.55)),
            
            # Topical Medications
            ('Antibiotic Ointment', 3.99, 7.99, 13.99, (0.48, 0.58)),
            ('Hydrocortisone Cream', 4.49, 8.99, 14.99, (0.45, 0.55)),
            ('Anti-Fungal Cream', 4.99, 9.99, 16.99, (0.48, 0.58)),
            ('Calamine Lotion', 3.49, 6.99, 11.99, (0.50, 0.60)),
            
            # Eye and Ear Care
            ('Eye Drops', 4.99, 9.99, 16.99, (0.48, 0.58)),
            ('Ear Drops', 5.49, 10.99, 17.99, (0.45, 0.55)),
            ('Contact Lens Solution', 6.99, 13.99, 21.99, (0.48, 0.58)),
            
            # Sleep and Energy
            ('Sleep Aid', 5.99, 11.99, 18.99, (0.48, 0.58)),
            ('Melatonin', 4.99, 9.99, 16.99, (0.45, 0.55)),
            ('Energy Supplements', 7.99, 15.99, 24.99, (0.48, 0.58))
        ],
        'Personal Care': [
            ('Toothpaste', 1.99, 3.99, 7.99, (0.50, 0.60)),
            ('Shampoo', 2.99, 5.99, 12.99, (0.52, 0.62)),
            ('Hand Sanitizer', 1.49, 2.99, 6.99, (0.55, 0.65)),
            ('Body Lotion', 2.49, 4.99, 9.99, (0.52, 0.62)),
            ('Sunscreen SPF 50', 4.49, 8.99, 15.99, (0.48, 0.58)),
            ('Facial Cleanser', 3.49, 6.99, 13.99, (0.50, 0.60))
        ],
        'Health Supplies': [
            ('Bandages', 2.49, 4.99, 8.99, (0.52, 0.62)),
            ('Vitamins', 6.49, 12.99, 24.99, (0.48, 0.58)),
            ('First Aid Kit', 7.99, 15.99, 29.99, (0.45, 0.55)),
            ('Blood Pressure Monitor', 17.99, 35.99, 59.99, (0.42, 0.52)),
            ('Thermometer', 4.49, 8.99, 16.99, (0.45, 0.55))
        ]
    }

    # Brand dictionaries for expanded products
    otc_brands = {
        'Acetaminophen 500mg': ['Tylenol', 'Panadol'],
        'Ibuprofen 200mg': ['Advil', 'Motrin'],
        'Naproxen 220mg': ['Aleve', 'Naprosyn'],
        'Aspirin 325mg': ['Bayer', 'St. Joseph'],
        'Cold & Flu Relief': ['DayQuil', 'Theraflu'],
        'Allergy Relief': ['Claritin', 'Allegra'],
        'Loratadine 10mg': ['Claritin', 'Generic Loratadine'],
        'Cetirizine 10mg': ['Zyrtec', 'Generic Cetirizine'],
        'Nasal Decongestant': ['Sudafed', 'Afrin'],
        'Cough Suppressant': ['Robitussin', 'Delsym'],
        'Throat Lozenges': ['Halls', 'Ricola'],
        'Sinus Relief': ['Mucinex', 'Advil Sinus'],
        'Antacid Tablets': ['TUMS', 'Rolaids'],
        'Acid Reducer': ['Prilosec', 'Zantac'],
        'Anti-Diarrheal': ['Imodium', 'Generic Loperamide'],
        'Fiber Supplement': ['Metamucil', 'Benefiber'],
        'Probiotics': ['Culturelle', 'Align'],
        'Antibiotic Ointment': ['Neosporin', 'Polysporin'],
        'Hydrocortisone Cream': ['Cortizone-10', 'Generic Hydrocortisone'],
        'Anti-Fungal Cream': ['Lotrimin', 'Tinactin'],
        'Calamine Lotion': ['Caladryl', 'Aveeno Calamine'],
        'Eye Drops': ['Visine', 'Refresh'],
        'Ear Drops': ['Debrox', 'Swim-EAR'],
        'Contact Lens Solution': ['Opti-Free', 'Biotrue'],
        'Sleep Aid': ['Unisom', 'ZzzQuil'],
        'Melatonin': ['Natrol', 'Nature Made'],
        'Energy Supplements': ['5-Hour Energy', 'Emergen-C']
    }

    personal_care_brands = {
        'Toothpaste': ['Colgate', 'Crest'],
        'Shampoo': ['Pantene', 'Head & Shoulders'],
        'Hand Sanitizer': ['Purell', 'Germ-X'],
        'Body Lotion': ['Nivea', 'Vaseline'],
        'Sunscreen SPF 50': ['Neutrogena', 'Banana Boat'],
        'Facial Cleanser': ['Cetaphil', 'CeraVe']
    }

    health_supplies_brands = {
        'Bandages': ['Band-Aid', 'Curad'],
        'Vitamins': ['Centrum', 'Nature Made'],
        'First Aid Kit': ['Johnson & Johnson', 'Swiss Safe'],
        'Blood Pressure Monitor': ['Omron', 'Welch Allyn'],
        'Thermometer': ['Braun', 'Vicks']
    }

    # Expand product list to include variants
    expanded_products = []
    for category, products_list in products.items():
        for product_info in products_list:
            product_name, base_cost, min_price, max_price, margin_range = product_info

            # Add base product
            expanded_products.append((product_name, base_cost, min_price, max_price, margin_range))

            # Add brand variants
            # Add brand or variant-specific products
            if category == "OTC Medication" and product_name in otc_brands:
                expanded_products.extend([
                    (f"{product_name} ({brand})", base_cost, min_price, max_price, margin_range)
                    for brand in otc_brands[product_name]
                ])
            elif category == "Personal Care" and product_name in personal_care_brands:
                expanded_products.extend([
                    (f"{product_name} ({brand})", base_cost, min_price, max_price, margin_range)
                    for brand in personal_care_brands[product_name]
                ])
            elif category == "Health Supplies" and product_name in health_supplies_brands:
                expanded_products.extend([
                    (f"{product_name} ({brand})", base_cost, min_price, max_price, margin_range)
                    for brand in health_supplies_brands[product_name]
                ])
            elif category == "Prescription":
                weights = ['10mg', '25mg', '50mg', '100mg']
                expanded_products.extend([
                    (f"{product_name.split()[0]} {weight}", base_cost, min_price, max_price, margin_range)
                    for weight in weights
                ])


    # Generate sales data
    data = []
    for date in dates:
        product_info = random.choice(expanded_products)
        product_name, base_cost, min_price, max_price, margin_range = product_info

        # Generate quantity
        quantity = random.choice([1, 2, 3])

        # Calculate price
        unit_price = round(random.uniform(min_price, max_price), 2)
        total_price = round(unit_price * quantity, 2)

        # Calculate cost and profit
        total_cost = round(base_cost * quantity, 2)
        gross_profit = round(total_price - total_cost, 2)
        profit_margin = round((gross_profit / total_price) * 100, 2)

        # Generate random payment method
        payment_method = random.choice(['Credit Card', 'Debit Card', 'Cash', 'Insurance'])

        # Generate customer and transaction IDs
        customer_id = f"CUST{random.randint(1, 50000):05d}"
        transaction_id = f"TXN{len(data):06d}"

        data.append({
            'transaction_id': transaction_id,
            'date': date,
            'customer_id': customer_id,
            'category': category,
            'product': product_name,
            'quantity': quantity,
            'unit_price': unit_price,
            'total_price': total_price,
            'total_cost': total_cost,
            'gross_profit': gross_profit,
            'profit_margin': profit_margin,
            'payment_method': payment_method
        })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Export to Excel
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"Data exported to {output_path}!")

# Run the function
generate_pharmacy_sales()
