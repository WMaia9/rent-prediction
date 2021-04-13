# pegar os dados do site zap Imóveis

import os
import settings
import pandas as pd
import zapimoveis_scraper as zap


"""
Função para puxar os dados no site Zap Moveis
E salva-los em um arquivo csv
"""
def web_scraper(local, x):
    tipo = ['casas', 'apartamentos']

    for t in tipo:
        result = zap.search(localization=local, num_pages=x, tipo=t)

        ofertas = len(result)

        df = pd.DataFrame(
            [],
            columns=[
                'Address',
                'Zone',
                'Type',
                'Total Area',
                'Bathrooms',
                'Bedrooms',
                'Vacancies',
                'Price',
                'Description',
                'Link'
            ],
        )

        for i in range(ofertas):
            df = df.append(
                {
                    'Address': result[i].address,
                    'Zone': local,
                    'Type': t,
                    'Total Area': result[i].total_area_m2,
                    'Bathrooms': result[i].bathrooms,
                    'Bedrooms': result[i].bedrooms,
                    'Vacancies': result[i].vacancies,
                    'Price': result[i].price,
                    'Description': result[i].description,
                    'Link': result[i].link
                },
                ignore_index=True,
            )

        df.to_csv(os.path.join(settings.dir_dados, '{}.csv'.format(local)), mode='a', index=False, header=False,
                  sep=',')


if __name__ == "__main__":
    #Localizada que será puxado os dados
    local = ['sp+sao-paulo+zona-oeste', 'sp+sao-paulo+zona-norte',
             'sp+sao-paulo+zona-sul', 'sp+sao-paulo+zona-leste',
             'sp+sao-paulo+centro']


    #Número de páginas de dados que são puxados
    x = int(input('number of pages: '))

    for i in local:
        web_scraper(i, x)
