;this is a example on how to calculate model/obs trend in our study 
;********************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"  
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"  
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl" 
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
;********************************************************
begin

;read near surface temperature and exract the time period we want (1970-2014) 

;month to annual 
can_tas = month_to_annual(f_Z3(time|:,ncl_join|:,lat|:,lon|:), 1)

;preparation for area weighting 
rad    = 4.0*atan(1.0)/180.0

;global mean near-surface temperature (GMST) 
cantas_ave = wgt_areaave_Wrap(can_tas, cos(lat*rad), 1.0, 1)

;calculate trend based on GMST
cetas_trend = dtrend(cantas_ave(ncl_join|:,year|:),True)
  
; ce_trend=new(6,"float")
trend= cetas_trend@slope

end
