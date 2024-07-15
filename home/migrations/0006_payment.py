# Generated by Django 5.0.1 on 2024-04-08 12:55

import datetime
import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0005_userprofile_daily_carbs_and_more'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Payment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('service', models.CharField(default='esewa', max_length=100)),
                ('payment_date', models.DateField(default=datetime.datetime(2024, 4, 8, 12, 55, 31, 605622, tzinfo=datetime.timezone.utc))),
                ('amount', models.FloatField()),
                ('transaction_uuid', models.CharField(max_length=100)),
                ('status', models.BooleanField(default=False)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
