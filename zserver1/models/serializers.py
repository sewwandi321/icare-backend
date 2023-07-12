from rest_framework import serializers
from .models import lab_report, blisters, blister_user, predefined_precautions, prescription, tablets, tablets_user, lab_report_precautions


class lab_reportSerializer(serializers.ModelSerializer):
    class Meta:
        model = lab_report
        fields = '__all__'

class lab_report_precautionsSerializer(serializers.ModelSerializer):
    class Meta:
        model = lab_report_precautions
        fields = '__all__'

class blistersSerializer(serializers.ModelSerializer):
    class Meta:
        model = blisters
        fields = '__all__'


class blister_userSerializer(serializers.ModelSerializer):
    class Meta:
        model = blister_user
        fields = '__all__'

class predefined_precautionsSerializer(serializers.ModelSerializer):
    class Meta:
        model = predefined_precautions
        fields = '__all__'

class prescriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = prescription
        fields = '__all__'

class tabletsSerializer(serializers.ModelSerializer):
    class Meta:
        model = tablets
        fields = '__all__'

class tablets_userSerializer(serializers.ModelSerializer):
    class Meta:
        model = tablets_user
        fields = '__all__'