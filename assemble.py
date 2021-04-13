import os
import settings
import pandas as pd

# Header para os dados do Zap Moveis
HEADERS = {
    "sp": [
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
    ]
}


# Função para concatenar o header com os dados fornecidos e todos os dados que estão separados por regiões.
def concatenar(prefixo: object) -> object:
    files = os.listdir(settings.dir_dados)
    full = []
    for i in files:
        if not i.startswith(prefixo):
            continue
        data = pd.read_csv(os.path.join(settings.dir_dados, i), sep=',', header=None, names=HEADERS[prefixo],
                           index_col=False)
        full.append(data)

    full = pd.concat(full, axis=0)
    full.to_csv(os.path.join(settings.dir_processos, "{}.csv".format(prefixo)), index=False)


if __name__ == "__main__":
    concatenar('sp')
