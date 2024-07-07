import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt
from pytrends.request import TrendReq
import pandas as pd


def regional(keyword):
    # Initialize pytrends
    pytrends = TrendReq(hl='en-US', tz=360)

    # Build the payload
    pytrends.build_payload([keyword], cat=0, timeframe='today 12-m', geo='IN')  # Analyzing trends over the past 12 months in the US

    # Get interest by region
    interest_by_region_df = pytrends.interest_by_region(resolution='REGION', inc_low_vol=True, inc_geo_code=False)

    # Display the DataFrame
    print(interest_by_region_df.sort_values(by=keyword, ascending=False))


    # Plotting (optional)

    interest_by_region_df.sort_values(by=keyword, ascending=True).plot(kind='barh', figsize=(12, 8), color='#86bf91', zorder=2, width=0.85)
    plt.title(f'Interest by Region for {keyword}')
    plt.xlabel('Relative Interest')
    plt.ylabel('Region')
    plt.savefig("static/img/region.png")
    plt.close()


