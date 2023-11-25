import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from babel.numbers import format_currency

sns.set(style="dark")

#Def Tren Penjualan
def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule="D", on="Date").agg(
        {"index": "nunique", "Amount": "sum"}
    )

    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(
        columns={"index": "order_count", "Amount": "revenue"}, inplace=True
    )
    
    return daily_orders_df

#Def Persentase Status
def persentase_status_shipping_df(df):
    total_status = len(df)
    
    status_shipping = [
        'Cancelled',
        'Shipped',
        'Shipped - Returned to Seller',
        'Pending',
        'Shipping',
        'Pending - Waiting for Pick Up',
        'Shipped - Damaged',
        'Shipped - Delivered to Buyer',
        'Shipped - Lost in Transit',
        'Shipped - Out for Delivery',
        'Shipped - Picked Up',
        'Shipped - Rejected by Buyer',
        'Shipped - Returning to Seller'
    ]
    
    persentase_status = {}
    
    for status in status_shipping:
        count_status = sum(df['Status'].str.lower() == status.lower())
        persentase = (count_status / total_status) * 100
        persentase_status[status] = persentase
    
    return persentase_status
  
#def fulfiment    
def banyak_Fulfilment_df(df):
    banyak_Fulfilment = df.groupby(by="Fulfilment")["index"].nunique().sort_values(ascending=False).reset_index()
    
    return banyak_Fulfilment

#def sales channel
def banyak_SalesChannel_df(df):
    banyak_SalesChannel = df.groupby(by="Sales Channel ").index.nunique().sort_values(ascending=False).reset_index()
    
    return banyak_SalesChannel

def banyak_service_pengiriman_df(df):
    banyak_service = df.groupby(by="ship-service-level").index.nunique().sort_values(ascending=False).reset_index()
    
    return banyak_service

#Def Best & Worst Performing Product
def create_sum_order_items_df(df):
    sum_order_items_df = (
        df.groupby("Category")
        .Qty.sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    return sum_order_items_df

#Def Size
def create_sum_order_size_df(df):
    sum_order_size_df = (
        df.groupby("Size")
        .Qty.sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    return sum_order_size_df

#Def Kurir status
def persentase_courir_status(df):
    jumlah_status_cancelled = len(df[df['Courier Status'] == 'Cancelled'])
    jumlah_status_shipped = len(df[df['Courier Status'] == 'Shipped'])
    jumlah_status_unshipped = len(df[df['Courier Status'] == 'Unshipped'])

    total_persen = len(df)

    persentase_cancelled = (jumlah_status_cancelled / total_persen) * 100
    persentase_shipped = (jumlah_status_shipped / total_persen) * 100
    persentase_unshipped = (jumlah_status_unshipped / total_persen) * 100

    persentase_cancelled_formatted = "{:.2f}".format(persentase_cancelled)
    persentase_shipped_formatted = "{:.2f}".format(persentase_shipped)
    persentase_unshipped_formatted = "{:.2f}".format(persentase_unshipped)

    hasil_persentase_courir = {
        'Cancelled': persentase_cancelled_formatted,
        'Shipped': persentase_shipped_formatted,
        'Unshipped': persentase_unshipped_formatted
    }

    return hasil_persentase_courir

#def penjualan terbanyak
def hitung_top_selling_products(df):
    df['Date'] = pd.to_datetime(df['Date'])

    df['Month'] = df['Date'].dt.to_period('M')

    monthly_sales = df.groupby(['Month', 'Category']).agg({'Qty': 'sum'}).reset_index()

    max_qty_indices = monthly_sales.groupby('Month')['Qty'].idxmax()

    top_selling_products = monthly_sales.loc[max_qty_indices]

    top_selling_products['Month'] = top_selling_products['Month'].dt.strftime('%B')

    return top_selling_products

#def perbandingan fifulment dan status courir
def analisis_filfulment_status(df):
    status_cancelled = df[df["Courier Status"] == "Cancelled"].groupby(by="Fulfilment")["index"].nunique()

    status_shipped = df[df["Courier Status"] == "Shipped"].groupby(by="Fulfilment")["index"].nunique()

    status_unshipped = df[df["Courier Status"] == "Unshipped"].groupby(by="Fulfilment")["index"].nunique()

    hasil_analisis = pd.DataFrame({
        "Status Cancelled": status_cancelled,
        "Status Shipped": status_shipped,
        "Status Unshipped": status_unshipped
    })
    
    return hasil_analisis

#def ship_city
def total_ship_city_df(df):
    ship_city_df = df.groupby(by="ship-city").index.nunique().sort_values(ascending=False).reset_index().head(10)
    
    return ship_city_df

#def ship_state
def total_ship_state_df(df):
    ship_state_df = df.groupby(by="ship-state").index.nunique().sort_values(ascending=False).reset_index().head(10)
    
    return ship_state_df

#def B2B
def total_b2b_df(df):
    b2b_df = df.groupby(by="B2B").index.nunique().sort_values(ascending=False).reset_index()
    
    b2b_df = df.groupby('B2B').size().reset_index(name='Count')
    
    return b2b_df

#def b2b persentase
def persetase_b2b_df(df):
    jumlah_b2b = len(df[df['B2B'] == 0])
    jumlah_b2bT = len(df[df['B2B'] == 1])

    total_persentase = len(df)

    persentase_b2b = (jumlah_b2b / total_persentase) * 100
    persentase_b2bT = (jumlah_b2bT / total_persentase) * 100

    persentase_b2b_formatted = "{:.2f}".format(persentase_b2b)
    persentase_b2bT_formatted = "{:.2f}".format(persentase_b2bT)
    
    hasil_persentase_b2b = {
        'B2B False' : persentase_b2b_formatted,
        'B2B True' : persentase_b2bT_formatted
    }
    
    return hasil_persentase_b2b

# st.set_option('deprecation.showPyplotGlobalUse', False)

sales_df = pd.read_csv("E-CommerceSalesAmazon.csv")

#set tanggal dari sampe kapan
datetime_columns = ["Date", "Date"]
sales_df.sort_values(by="Date", inplace=True)
sales_df.reset_index(inplace=True)

for column in datetime_columns:
    sales_df[column] = pd.to_datetime(sales_df[column])

min_date = sales_df["Date"].min()
max_date = sales_df["Date"].max()    

with st.sidebar:
    start_date, end_date = st.date_input(
            label="Rentang Waktu",
            min_value=min_date,
            max_value=max_date,
            value=[min_date, max_date],
        )
    
    options = st.multiselect('Hasil Analisis Bisnis :',
                             ['Sales Trends', 'Delivery Status', 'Daily Courier Service' ,'Daily Fulfilment','Daily Sales Channel','Courier Delivery Status', 'Best & Worst Performing Category', 'Most & least selling Size' , 'Purchases and Courier Delivery Status' ,'Correlation between Category and Size', 'Best-Selling Product Each Month', 'Top 10 Ship Cities', 'Top 10 Ship State', 'B2B'
                            ])
    
    
main_df = sales_df[
    (sales_df["Date"] >= str(start_date)) & (sales_df["Date"] <= str(end_date))
]

daily_orders_df = create_daily_orders_df(main_df)
total_status = persentase_status_shipping_df(main_df)
banyak_Fulfilment = banyak_Fulfilment_df(main_df)
banyak_SalesChannel = banyak_SalesChannel_df(main_df)
banyak_service = banyak_service_pengiriman_df(main_df)
sum_order_items_df = create_sum_order_items_df(main_df)
sum_order_size_df = create_sum_order_size_df(main_df)
hasil_persentase_courir = persentase_courir_status(main_df)
top_selling_products = hitung_top_selling_products(main_df)
hasil_analisis = analisis_filfulment_status(main_df)
ship_city_df = total_ship_city_df(main_df)
ship_state_df = total_ship_state_df(main_df)
b2b_df = total_b2b_df(main_df)
hasil_persentase_b2b = persetase_b2b_df(main_df)
#sampe sini code tanggal

if not options:
    st.markdown(
        """
        <h1 style="text-align: center;">Amazon E-Commerce Sales Analyze Dashboard</h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
    """
    <h4 style="text-align: center;">The results of the analysis of the E-Commerce Sales Dataset from Amazon</h4>
    """,
    unsafe_allow_html=True
    )

else:
    for option in options:
        
        if option == 'Sales Trends':
        #Visual Tren Penjualan
            st.subheader("Sales Trends")

            col1, col2 = st.columns(2)
            with col1:
                total_orders = daily_orders_df.order_count.sum()
                st.metric("Total orders", value=total_orders)

            with col2:
                total_revenue = format_currency(
                daily_orders_df.revenue.sum(), "IN ", locale="en_IN"
                )
                st.metric("Total Revenue", value=total_revenue)

            st.subheader("Daily Orders")
            plt.figure(figsize=(10, 5))
            plt.plot(
                daily_orders_df["Date"],
                daily_orders_df["order_count"],
                marker='o',
                linewidth=2,
                color="orange"
            )
            plt.title("Many orders per month", loc="center", fontsize=20)
            plt.xticks(rotation=45, fontsize=10)
            plt.yticks(fontsize=10)
            st.pyplot(plt)

            st.subheader("Daily Revenue")

            plt.figure(figsize=(10, 5))
            plt.plot(
                daily_orders_df["Date"],
                daily_orders_df["revenue"],
                marker='o',
                linewidth=2,
                color="#72BCD4"
            )
            plt.title("Daily Revenue", loc="center", fontsize=20)
            plt.xticks(rotation=45, fontsize=10)
            plt.yticks(fontsize=10)
            st.pyplot(plt)
        #Sampe SIni

        elif option == 'Delivery Status':
        #Visual Status pengiriman
            st.subheader("Delivery Status")
            labels = ['Status Cancelled', 'Status Shipping', 'Status Shipped', 'Status Returned To Seller', 'Status Pending', 'Status Waiting Pick Up', 'Status Damaged', 'Status Delivered to Buyer',
                    'Status Lost In Transit', 'Status Out Of Delivery', 'Status Pick Up', 'Status Rejected Buyer', 'Status Returning To seller']
            status_shipping = list(total_status.keys())
            colors = ['#66b3ff', '#ff9999', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c2f0c2', '#ffb366', '#e6f2ff', '#ffb3b3', '#c2f0f0', '#ff6666', '#ff6666']
            explode = (0,) + (0,) * (len(labels) - 1)
            # Menyiapkan data untuk diagram pie
            sizes = [total_status[status] for status in status_shipping]

            # Menangani nilai di bawah 0%
            sizes = [max(0, val) for val in sizes]

            # Menambahkan kondisi untuk menangani nilai "Lain-lain"
            total_sizes = sum(sizes)
            threshold_percentage = 1  # Atur ambang batas persentase di bawah 1% sebagai "Lain-lain"
            other_indices = [i for i, val in enumerate(sizes) if val / total_sizes < threshold_percentage / 100]
            other_size = sum(sizes[i] for i in other_indices)

            # Mengganti nilai "Lain-lain" dengan label "Lain-lain"
            for i in other_indices:
                status_shipping[i] = 'Lain-lain'

            # Menghapus nilai-nilai "Lain-lain" dari sizes dan labels
            sizes = [sizes[i] for i in range(len(sizes)) if i not in other_indices]
            status_shipping = [status for i, status in enumerate(status_shipping) if i not in other_indices]

            # Membuat diagram pie
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.pie(sizes + [other_size], explode=[0] * len(sizes) + [0.0], labels=status_shipping + ['Dan lain-lain'], colors=colors, autopct='%1.2f%%', shadow=True, startangle=140)

            # Menambahkan legenda
            ax.legend(status_shipping + ['Dan Lain-lain'], loc="upper left", bbox_to_anchor=(1, 0.5))

            plt.title("Percentage of Delivery Status")
            plt.axis('equal')
            st.pyplot(fig)
        #Sampe Sini

        elif option == 'Daily Fulfilment':
        #Visual Banyak Fulfilment
            st.subheader("Daily Fulfilment")
            col1, col2 = st.columns(2)
            banyak_Fulfilment_df_result = banyak_Fulfilment_df(main_df)
            columns = [col1, col2]

            for i, row in banyak_Fulfilment_df_result.iterrows():
                fulfilment, value = row["Fulfilment"], row["index"]
                col_to_use = col1 if i % 2 == 0 else col2  # Choose the column based on the value of i
                with col_to_use:
                    st.metric(fulfilment, value)
            plt.figure(figsize=(10, 6))
            plt.bar(banyak_Fulfilment["Fulfilment"], banyak_Fulfilment["index"], color='skyblue')
            plt.title("Count of Unique Indices by Fulfilment")
            plt.xlabel("Fulfilment")
            plt.ylabel("Count of Unique Indices")
            plt.xticks(rotation=0, ha='right')
            st.pyplot(plt)
        #Sampe Sini

        elif option == 'Daily Sales Channel':
        #Visual Banyak Fulfilment
            st.subheader("Daily Sales Channel")
            col1, col2 = st.columns(2)
            banyak_SalesChannel_df_result = banyak_SalesChannel_df(main_df)
            columns = [col1, col2] 

            for i, row in banyak_SalesChannel_df_result.iterrows():
                sales_channel, value = row["Sales Channel "], row["index"]
                col_to_use = col1 if i % 2 == 0 else col2  # Choose the column based on the value of i
                with col_to_use:
                    st.metric(sales_channel, value)
                        
            plt.figure(figsize=(10, 6))
            plt.bar(banyak_SalesChannel["Sales Channel "], banyak_SalesChannel["index"], color='red')
            plt.title("Count of Unique Indices by Sales Channel")
            plt.xlabel("Sales Channel")
            plt.ylabel("Count of Unique Indices")
            plt.xticks(rotation=0, ha='right')
            st.pyplot(plt)
        #Sampe Sini

        elif option == 'Daily Courier Service':
        #Visual Ekspedisi kurir
            st.subheader("Courier Service")
            col1, col2 = st.columns(2)
            banyak_service_pengiriman_df_result = banyak_service_pengiriman_df(main_df)
            columns = [col1, col2]

            for i, row in banyak_service_pengiriman_df_result.iterrows():
                sales_channel, value = row["ship-service-level"], row["index"]
                col_to_use = col1 if i % 2 == 0 else col2  # Choose the column based on the value of i
                with col_to_use:
                    st.metric(sales_channel, value)
                    
            plt.figure(figsize=(10, 6))
            plt.bar(banyak_service["ship-service-level"], banyak_service["index"], color='green')
            plt.title("Count Courier Service")
            plt.xlabel("Service Ship Lavel")
            plt.ylabel("Count of Service Ship Lavel")
            plt.xticks(rotation=0, ha='right')
            st.pyplot(plt)
        #Sampe Sini

        elif option == 'Best & Worst Performing Category':
        #Category paling laris
            st.subheader("Best & Worst Performing Category")
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))
            colors = ["#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9"]

            sns.barplot(
                x="Qty",
                y="Category",
                data=sum_order_items_df.head(4),
                palette=colors,
                ax=ax[0],
            )
            ax[0].set_ylabel(None)
            ax[0].set_xlabel("Number of Sales", fontsize=30)
            ax[0].set_title("Best Performing Category", loc="center", fontsize=50)
            ax[0].tick_params(axis="y", labelsize=35)
            ax[0].tick_params(axis="x", labelsize=30)

            sns.barplot(
                x="Qty",
                y="Category",
                data=sum_order_items_df.sort_values(by="Qty", ascending=True).head(4),
                palette=colors,
                ax=ax[1],
            )
            ax[1].set_ylabel(None)
            ax[1].set_xlabel("Number of Sales", fontsize=30)
            ax[1].invert_xaxis()
            ax[1].yaxis.set_label_position("right")
            ax[1].yaxis.tick_right()
            ax[1].set_title("Worst Performing Category", loc="center", fontsize=50)
            ax[1].tick_params(axis="y", labelsize=35)
            ax[1].tick_params(axis="x", labelsize=30)

            st.pyplot(fig)
        #sampe Sini

        elif option == 'Most & least selling Size':
        #Visualisasi Ukuran
            st.subheader("Most & least selling Size")
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))
            colors = ["#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9"]
            sns.barplot(
                x="Qty",
                y="Size",
                data=sum_order_size_df.head(5),
                palette=colors,
                ax=ax[0],
            )
            ax[0].set_ylabel(None)
            ax[0].set_xlabel("Number of Sales", fontsize=30)
            ax[0].set_title("The most popular Size", loc="center", fontsize=50)
            ax[0].tick_params(axis="y", labelsize=35)
            ax[0].tick_params(axis="x", labelsize=30)

            sns.barplot(
                x="Qty",
                y="Size",
                data=sum_order_size_df.sort_values(by="Qty", ascending=True).head(5),
                palette=colors,
                ax=ax[1],
            )
            ax[1].set_ylabel(None)
            ax[1].set_xlabel("Number of Sales", fontsize=30)
            ax[1].invert_xaxis()
            ax[1].yaxis.set_label_position("right")
            ax[1].yaxis.tick_right()
            ax[1].set_title("The least selling Size", loc="center", fontsize=50)
            ax[1].tick_params(axis="y", labelsize=35)
            ax[1].tick_params(axis="x", labelsize=30)

            st.pyplot(fig)
        #sampe Sini

        elif option == 'Correlation between Category and Size':
        #Visualisasi Korelasi category vs size
            st.subheader("The Correlation between Category and Size")
            plt.figure(figsize=(10, 6))
            heatmap_data = pd.pivot_table(main_df, values='Qty', index='Category', columns='Size', aggfunc='sum', fill_value=0)
            sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt='g')
            plt.title('The Correlation between Category and Size')
            st.pyplot(plt)
            #sampe sini

        elif option == 'Courier Delivery Status':
            #Visual persentase courir status
            labels = list(hasil_persentase_courir.keys())
            sizes = [float(value) for value in hasil_persentase_courir.values()]
            colors = ['#66b3ff', '#ff9999', 'green']
            explode = (0, 0, 0)
            # Membuat diagram pie
            st.subheader("Courier Delivery Status")
            plt.figure(figsize=(7, 7))
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%', shadow=True, startangle=140)
            plt.title("Courier Delivery Status")
            plt.axis('equal')

            st.pyplot(plt)
        #sampe sini

        elif option == 'Best-Selling Product Each Month':
            #Produk Paling Laris Setiap Bulan
            st.subheader("Best-Selling Product Each Month")
            # Assuming 'Month' column contains month names like 'January', 'February', etc.
            top_selling_products['Month'] = pd.to_datetime(top_selling_products['Month'], format='%B').dt.month

            # Sort the DataFrame by the 'Month' column
            top_selling_products = top_selling_products.sort_values(by='Month')

            plt.figure(figsize=(10, 6))
            for category in top_selling_products['Category'].unique():
                category_data = top_selling_products[top_selling_products['Category'] == category]
                plt.bar(category_data['Month'], category_data['Qty'], label=f'Merk {category}', alpha=0.7)  # Added alpha for transparency

            plt.title('Best-Selling Product Each Month')
            plt.xlabel('Moth')
            plt.ylabel('Total Sales')
            plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Customize the x-axis ticks
            plt.legend()
            st.pyplot(plt)
        #sampe Sini

        elif option == 'Purchases and Courier Delivery Status':
            #Perbandingan Pembelian dan status pengiriman kurir
            st.subheader("Comparison of Purchases and Courier Delivery Status")
            status = hasil_analisis.index
            status_cancelled = hasil_analisis["Status Cancelled"]
            status_shipped = hasil_analisis["Status Shipped"]
            status_unshipped = hasil_analisis["Status Unshipped"]

            bar_width = 0.2

            x = range(len(status))

            plt.figure(figsize=(10, 6))
            plt.barh([i + bar_width for i in x], status_cancelled, bar_width, label="Status Cancelled", color='red')
            plt.barh(x, status_shipped, bar_width, label="Status Shipped", color='green')
            plt.barh([i - bar_width for i in x], status_unshipped, bar_width, label="Status Unshipped", color='gray')

            plt.yticks(x, status)
            plt.xlabel("Jumlah Pengiriman")
            plt.title("Comparison of Purchases and Courier Delivery Status")

            plt.legend()
            st.pyplot(plt)
        #sampe sini

        elif option == 'Top 10 Ship Cities':
            #Visual Top 10 Ship Cities by Index Count
            st.subheader("Top 10 Ship Cities by Index Count")
            sns.set(style="whitegrid")

            plt.figure(figsize=(12, 8))
            bar_plot = sns.barplot(x='ship-city', y='index', data=ship_city_df, palette='viridis')

            plt.title('Top 10 Ship Cities by Index Count')
            plt.xlabel('Ship City')
            plt.ylabel('Index Count')

            for p in bar_plot.patches:
                bar_plot.annotate(format(p.get_height(), '.0f'),
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha = 'center', va = 'center',
                                xytext = (0, 9),
                                textcoords = 'offset points')
            bar_plot.set_xticklabels(bar_plot.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(plt)
        #sampe sini

        elif option == 'Top 10 Ship State':
        #Top 10 Ship State by Index Count
            st.subheader("Top 10 Ship State by Index Count")
            sns.set(style="whitegrid")

            plt.figure(figsize=(12, 8))
            bar_plot = sns.barplot(x='ship-state', y='index', data=ship_state_df, palette='viridis')

            plt.title('Top 10 Ship State by Index Count')
            plt.xlabel('Ship State')
            plt.ylabel('Index Count')

            for p in bar_plot.patches:
                bar_plot.annotate(format(p.get_height(), '.0f'),
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha = 'center', va = 'center',
                                xytext = (0, 9),
                                textcoords = 'offset points')
            bar_plot.set_xticklabels(bar_plot.get_xticklabels(), rotation=45, ha='right')

            st.pyplot(plt)
        #sampe Sini

        elif option == 'B2B':
            # Jumlah Occurrences B2B
            st.subheader("The Number of B2B Occurrences")
            plt.figure(figsize=(8, 6))
            plt.bar(b2b_df['B2B'].astype(str), b2b_df['Count'], color=['blue', 'orange'])
            plt.xlabel('Tipe Bisnis')
            plt.ylabel('Jumlah')
            plt.title('The Number of B2B Occurrences')
            st.pyplot(plt)
            #sampe sini

            #Persentase B2B
            st.subheader("B2B Percentage")
            labels = list(hasil_persentase_b2b.keys())
            sizes = [float(value) for value in hasil_persentase_b2b.values()]
            colors = ['#66b3ff', '#ff9999']
            explode = (0, 0)
            #sampe sini

            # Membuat diagram pie
            plt.figure(figsize=(7, 7))
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%', shadow=True, startangle=140)
            plt.title("B2B Percentage")
            plt.axis('equal')

            st.pyplot(plt)
        #sampe sini

    #END
st.caption("Copyright (c) Muhammad Fikri Ramadhan")