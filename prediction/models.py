from django.db import models

# Create your models here.

class Studenttxt(models.Model):
    q_txt = models.CharField(max_length=200,
     default="@AmericanAir In car gng to DFW. Pulled over 1hr ago - very icy roads. On-hold with AA since 1hr. Can't reach arpt for AA2450. Wat 2 do"   
     )