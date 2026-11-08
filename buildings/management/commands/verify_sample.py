"""Exercise sample imagery through the API and real model."""
from django.core.management.base import BaseCommand
from django.test import Client
from shapely.geometry import Polygon


class Command(BaseCommand):
    help = "Verify real sample inference, polygons and result URLs."

    def handle(self, *args, **options):
        client = Client()
        r = client.post('/api/detect-changes/', data={
            'old_image_path': 'past_sample.tif',
            'new_image_path': 'current_sample (1).tif',
        }, content_type='application/json')
        assert r.status_code == 200, r.content
        result = r.json()
        assert result['new_building_count'] == len(result['detections']) > 0
        w, h = (result['image_dimensions'][k] for k in ('width', 'height'))
        for d in result['detections']:
            p = Polygon(d['polygon'])
            assert p.is_valid and p.area > 0
            assert d['polygon'][0] == d['polygon'][-1]
            assert all(0 <= x <= w and 0 <= y <= h for x, y in d['polygon'])
            assert d['area']['value'] > 0
        for url in [*result['preview_urls'].values(), result['result_url']]:
            response = client.get(url)
            assert response.status_code == 200
            response.close()
        self.stdout.write(f"SAMPLE VERIFIED: {result['new_building_count']} candidates; {result['alignment']['method']}")
        self.stdout.write(result['result_url'])
