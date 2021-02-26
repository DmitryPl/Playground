from typing import Optional

import requests
from loguru import logger


class HereGeocoder:
    __reverse = 'https://revgeocode.search.hereapi.com/v1/revgeocode'
    __geocoder = 'https://geocode.search.hereapi.com/v1/geocode'

    def __init__(self, api_key):
        self.__api_key = api_key

    def from_address(self, address: str, lang: str = 'ru-RU', postal_code=None) -> dict:
        params = {'q': address, 'lang': lang, 'countryCode': 'RUS', 'apiKey': self.__api_key}
        if postal_code is not None:
            params['postalCode'] = postal_code
        r = requests.get(self.__geocoder, params=params)
        logger.info(f'Here API, get from address, request: {r.status_code}, {r.reason}')
        return r.json()

    def from_postal_code(self, postal_code: str, lang: str = 'ru-RU') -> dict:
        params = {'q': postal_code, 'lang': lang, 'countryCode': 'RUS', 'apiKey': self.__api_key}
        r = requests.get(self.__geocoder, params=params)
        logger.info(f'Here API, get from postal code, request: {r.status_code}, {r.reason}')
        return r.json()

    def from_point(self, lat: float, lon: float, lang: str = 'ru-RU') -> dict:
        params = {'at': f'{lat},{lon}', 'lang': lang, 'apiKey': self.__api_key}
        r = requests.get(self.__reverse, params=params)
        logger.info(f'Here API, get from point, request: {r.status_code}, {r.reason}')
        return r.json()

    @staticmethod
    def get_point(result: dict) -> Optional[tuple[float, float]]:
        if 'items' not in result or not result['items'] or 'position' not in result['items'][0]:
            return None
        loc = result['items'][0]['position']
        return float(loc['lat']), float(loc['lng'])

    @staticmethod
    def get_address(result: dict) -> Optional[str]:
        if 'items' not in result or not result['items'] or 'title' not in result['items'][0]:
            return None
        return result['items'][0]['title']

    @staticmethod
    def get_city(result: dict) -> Optional[str]:
        if 'items' not in result or not result['items'] or 'city' not in result['items'][0]:
            return None
        return result['items'][0]['city']

    @staticmethod
    def get_state(result: dict) -> Optional[str]:
        if 'items' not in result or not result['items'] or 'state' not in result['items'][0]:
            return None
        return result['items'][0]['state']

    @staticmethod
    def get_postal_code(result: dict) -> Optional[str]:
        if 'items' not in result or not result['items'] or 'postalCode' not in result['items'][0]:
            return None
        return result['items'][0]['postalCode']
