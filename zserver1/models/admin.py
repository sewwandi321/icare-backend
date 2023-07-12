from django.contrib import admin
from .models import lab_report, blisters, blister_user, predefined_precautions, tablets, tablets_user, prescription, lab_report_precautions
# Register your models here.

admin.site.register(lab_report)
admin.site.register(blisters)
admin.site.register(blister_user)
admin.site.register(predefined_precautions)
admin.site.register(tablets)    
admin.site.register(tablets_user)
admin.site.register(prescription)
admin.site.register(lab_report_precautions)

