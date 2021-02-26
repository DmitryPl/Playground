from typing import Optional

import requests
from loguru import logger


class YandexGeocoder:
    __geocoder = 'https://geocode-maps.yandex.ru/1.x/'

    def __init__(self, api_key):
        self.__api_key = api_key

    def from_address(self, address: str) -> dict:
        params = {'format': 'json', 'apikey': self.__api_key, 'geocode': address}
        r = requests.get(self.__geocoder, params=params)
        logger.info(f'Yandex API, get from address, request: {r.status_code}, {r.reason}')
        return r.json()

    def from_point(self, lat: float, lon: float) -> dict:
        params = {'format': 'json', 'apikey': self.__api_key, 'geocode': f'{lat},{lon}', 'sco': 'latlong', 'results': 1}
        r = requests.get(self.__geocoder, params=params)
        logger.info(f'Yandex API, get from point, request: {r.status_code}, {r.reason}')
        return r.json()

    @staticmethod
    def get_point(result: dict) -> Optional[tuple[float, float]]:
        if 'response' not in result or not result['response']['featureMember']:
            return None
        loc = result['response']['featureMember'][0]['GeoObject']['Point']['pos'].split(' ')
        return float(loc[1]), float(loc[2])

    @staticmethod
    def get_address(result: dict) -> Optional[str]:
        if 'response' not in result or not result['response']['featureMember']:
            return None
        return result['response']['featureMember'][0]['GeoObject']['metaDataProperty']['GeocoderMetaData']['text']

    @staticmethod
    def get_city(result: dict) -> Optional[str]:
        if 'response' not in result or not result['response']['featureMember']:
            return None
        addr = result['response']['featureMember'][0]['GeoObject']['metaDataProperty']['GeocoderMetaData']['Address']
        return next(x for x in addr['Components'] if x['kind'] == 'locality')

    @staticmethod
    def get_state(result: dict) -> Optional[str]:
        if 'response' not in result or not result['response']['featureMember']:
            return None
        addr = result['response']['featureMember'][0]['GeoObject']['metaDataProperty']['GeocoderMetaData']['Address']
        return next(x for x in addr['Components'] if x['kind'] == 'province')

    @staticmethod
    def get_postal_code(result: dict) -> Optional[str]:
        if 'response' not in result or not result['response']['featureMember']:
            return None
        addr = result['response']['featureMember'][0]['GeoObject']['metaDataProperty']['GeocoderMetaData']['Address']
        return addr['postal_code']
