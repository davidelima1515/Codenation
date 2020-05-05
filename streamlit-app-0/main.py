import pandas as pd 
import streamlit as st 


def main():
    name = st.text_input('Qual o seu nome? ')
    if name:
        st.write('Bem-vindo(a), ', name,'!!!!')
        st.write('É um prazer recebê-lo(a) aqui.')
        carregando_arq = st.file_uploader('Entre com o arquivo com a qual deseja trabalhar: ', type='csv')
        if carregando_arq:
            leitor = pd.read_csv(carregando_arq)
            st.write(leitor.head(10))
            st.sidebar.text('O QUE VOCÊ QUER VER AGORA? ')
            st.sidebar.selectbox('ESCOLHA DENTRE AS OPÇÕES ABAIXO: ', ('PRÉ-PROCESSAMENTO DE DADOS', 'ANALISE DE DADOS', 'REGRESSÃO'))
            if 'PRÉ-PROCESSAMENTO DE DADOS':
                st.header('PRÉ-PROCESSAMENTO DE DADOS!!!!')
                st.subheader('DADOS')
                st.write(leitor.isna().sum())

            
         
            


if __name__ == "__main__":
    main()