{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 176,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('desafio1.csv', usecols=['id','estado_residencia','pontuacao_credito'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "metadata": {},
   "outputs": [],
   "source": [
    "lista_de_estados = df['estado_residencia'].unique()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "metadata": {},
   "outputs": [],
   "source": [
    "dicionario_resposta = {}\n",
    "for iterador in lista_de_estados:\n",
    "    dicionario_resposta[iterador] = {\n",
    "        \n",
    "                \n",
    "        \"moda\":df[df['estado_residencia'] == iterador]['pontuacao_credito'].mode()[0],\n",
    "        \"mediana\": df[df['estado_residencia']==iterador]['pontuacao_credito'].median(),\n",
    "        \"media\": df[df['estado_residencia']==iterador]['pontuacao_credito'].mean(),\n",
    "        \"desvio_padrao\": df[df['estado_residencia'] == iterador]['pontuacao_credito'].std()\n",
    "    \n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'SC': {'moda': 850,\n",
       "  'mediana': 653.0,\n",
       "  'media': 649.5376527422563,\n",
       "  'desvio_padrao': 97.233492793433},\n",
       " 'RS': {'moda': 850,\n",
       "  'mediana': 650.0,\n",
       "  'media': 651.1051428571428,\n",
       "  'desvio_padrao': 95.13659841383574},\n",
       " 'PR': {'moda': 850,\n",
       "  'mediana': 650.0,\n",
       "  'media': 648.9612940496822,\n",
       "  'desvio_padrao': 98.60718591309758}}"
      ]
     },
     "execution_count": 180,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dicionario_resposta"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 181,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_resposta = pd.DataFrame(dicionario_resposta)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 182,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_resposta.to_json('submission.json')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.7.6 32-bit ('venv': virtualenv)",
   "language": "python",
   "name": "python37632bitvenvvirtualenv1182be4b28d84d5fbffa1001ff096bcc"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
