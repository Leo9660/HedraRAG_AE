source data.conf

cd evaluation/fig_15

bash HedraRAG_test.sh
bash FlashRAG_test.sh
bash LangChain_test.sh
python draw_fig.py
