from asyncio import get_event_loop
from typing import Optional

import orjson
import requests
from aiohttp import ClientSession
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

    def batch_geocode(self, addresses: list[str]) -> dict[str, tuple[float, float]]:
        async def runner():
            async with ClientSession() as session:
                return await self.download(session, addresses)

        return get_event_loop().run_until_complete(runner())

    async def download(self, session, addresses: list[str], batch=300) -> dict[str, Optional[tuple[float, float]]]:
        address_set = list(set(addresses))
        size = len(address_set)
        result = {}

        for start in range(0, size, batch):
            addr_slice = address_set[start: start + batch]
            result |= {addr: await self.__geocode(session, addr) for addr in addr_slice}

        return result

    async def __geocode(self, session, addr: str, lang: str = 'ru-RU') -> Optional[tuple[float, float]]:
        params = {'q': addr, 'lang': lang, 'countryCode': 'RUS', 'apiKey': self.__api_key}
        result = await self.__get(self.__geocoder, session, params)
        return self.get_point(result)

    @staticmethod
    async def __get(url: str, session: ClientSession, params: dict) -> dict:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f'Response error: {resp.status} - {resp.reason}')
            return orjson.loads(await resp.text('utf-8'))

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
        if 'items' not in result or not result['items']:
            return None
        return result['items'][0]['address']['city']

    @staticmethod
    def get_state(result: dict) -> Optional[str]:
        if 'items' not in result or not result['items']:
            return None
        return result['items'][0]['address']['state']

    @staticmethod
    def get_postal_code(result: dict) -> Optional[str]:
        if 'items' not in result or not result['items']:
            return None
        return result['items'][0]['address']['postalCode']
