from django.db import models

def upload_to_lab_report(instance, filename):
    return 'images/lab_report_{filename}'.format(filename=filename)

def upload_to_blister(instance, filename):
    return 'images/blister_{filename}'.format(filename=filename)

def upload_to_tablet(instance, filename):
    return 'images/tablet_{filename}'.format(filename=filename)

def upload_to_blister_user(instance, filename):
    return 'images/blister_user_{filename}'.format(filename=filename)

def upload_to_prescription(instance, filename):
    return 'images/prescription_{filename}'.format(filename=filename)

""""
--------------------------------------------------------------
name           | precaution       | description       | 

cholesterol    |                  |                   | 
suguars        | do not eat suguar| long description  |
blood_pressure |                  |                   |
"""

class blister_user(models.Model):
    user = models.ForeignKey('accounts.MyUser', on_delete=models.CASCADE)
    image_url = models.ImageField(upload_to=upload_to_blister_user, blank=True, null=True)
    pass

class tablets(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    side_effects = models.TextField()

class tablets_user(models.Model):
    name = models.CharField(max_length=100, blank=True, null=True)
    image_url = models.ImageField(upload_to=upload_to_tablet, blank=True, null=True)


class predefined_precautions(models.Model):
    # user = models.ForeignKey('accounts.MyUser', on_delete=models.CASCADE)
    medicine_name = models.CharField(max_length=100)
    medicine_details = models.TextField()
    common_side_effects = models.TextField()
    other_side_effects = models.TextField()
    severe_side_effects = models.TextField()


class lab_report(models.Model):
    user = models.ForeignKey('accounts.MyUser', on_delete=models.CASCADE)
    item1 = models.CharField(max_length=100, blank=True, null=True)
    item2 = models.CharField(max_length=100, blank=True, null=True)
    item3 = models.CharField(max_length=100, blank=True, null=True)
    item4 = models.CharField(max_length=100, blank=True, null=True)
    item5 = models.CharField(max_length=100, blank=True, null=True)
    item6 = models.CharField(max_length=100, blank=True, null=True)
    item7 = models.CharField(max_length=100, blank=True, null=True)
    item8 = models.CharField(max_length=100, blank=True, null=True)
    image_url = models.ImageField(upload_to=upload_to_lab_report, blank=True, null=True)

class lab_report_precautions(models.Model):
    name = models.CharField(max_length=100, blank=True, null=True)
    Level = models.CharField(max_length=100, blank=True, null=True)
    Precautions = models.TextField(blank=True, null=True)
# pass

class blisters(models.Model):
    name = models.CharField(max_length=100, blank=True, null=True)
    text = models.TextField()
    image_url = models.ImageField(upload_to=upload_to_blister, blank=True, null=True)
    # pass


class prescription(models.Model):
    user = models.ForeignKey('accounts.MyUser', on_delete=models.CASCADE)
    name = models.CharField(max_length=100, blank=True, null=True)
    doseage = models.FloatField(blank=True, null=True)
    image_url = models.ImageField(upload_to=upload_to_prescription, blank=True, null=True)
    # pass