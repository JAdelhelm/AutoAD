from django.urls import path

from . import views

app_name = "detection"

urlpatterns = [
    path("", views.index, name="index"),
    path("examples/<slug:key>/", views.load_example, name="example"),
    path("analyses/<uuid:pk>/", views.analysis_detail, name="analysis"),
    path("analyses/<uuid:pk>/configure/", views.configure, name="configure"),
    path("analyses/<uuid:pk>/check/", views.check, name="check"),
    path("analyses/<uuid:pk>/delete/", views.delete_analysis, name="delete_analysis"),
    path("results/<uuid:pk>/", views.scoring_detail, name="scoring"),
    path("results/<uuid:pk>/rows/<int:number>/", views.row_detail, name="row"),
    path("results/<uuid:pk>/report.csv", views.download_report, name="download"),
    path("results/<uuid:pk>/delete/", views.delete_scoring, name="delete_scoring"),
    path("assets/plotly.min.js", views.plotly_js, name="plotly_js"),
]
