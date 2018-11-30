gstore_data:

-Originally from trainv2 from Kaggle

-JSON columns were disambiguated and formatted into proper csv format with empty columns removed -> trainv2_clean.csv

-continuous data was then normalized -> trainv2_df.csv

-dataset was balanced so that 10% of samples had transactions, and transaction columns were simplified to binary class (bought or did not). also dropped referralLink and networkDomain columns as they had many categories (1000s) -> trainv2_10.csv
