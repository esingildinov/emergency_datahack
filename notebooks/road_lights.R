# Cloud_coverage
# Emergency DataHack
# Информация по покрытию облаков (ночному освещению) для дорог

# Install or load required dependencies
packages <- c("data.table", "stringi", "stringr", "lubridate", "Rcpp", "sf", "mapview", "cptcity", "geojsonio", "rgee")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
        install.packages(setdiff(packages, rownames(installed.packages())), quiet = T) 
}
lapply(packages, library, character.only = T)

library(sf)


# Геокодер дорорг
library(tidyverse)
geo <- read.csv("geo_imputed.csv")
geo1 <- geo%>% select(road_id, road_km, lat_geoc, lon_geoc)


# Сохраняем как shapefile

sfd = st_as_sf(geo1, coords=c("lon_geoc","lat_geoc"), crs=4088)

# Создаем буфер вокруг дорог
roads_buffer <- st_buffer(sfd, 500)

roads_buffer <- st_transform(roads_buffer, crs = 4326)

st_write(roads_buffer, "geo_roads_buffer.shp", delete_layer = T, layer_options = "ENCODING = UTF-8")


# # Authenticate to Google Earth Engine
# (it will return an error the first time, but it is expected)
ee_Initialize(display = T)

# Светимость

NOAA_ligth <- ee$ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG") %>% 
        ee$ImageCollection$filterDate("2017-01-01", "2020-12-31") %>% 
        ee$ImageCollection$map(function(x) x$select("cf_cvg"))
#, "avg_rad"


# Из-за объема и низкой скорости обработки базы данных, делим её на 5 частей
manyroads_light <- ee_extract(x = NOAA_ligth,
                              y = roads_buffer [1:500,],
                              fun = ee$Reducer$mean(),
                              via = "drive", # This will not work with large-scale exports
                              scale = 500,
                              sf = F)

manyroads_light1 <- ee_extract(x = NOAA_ligth,
                               y = roads_buffer [501:1000,],
                               fun = ee$Reducer$mean(),
                               via = "drive", # This will not work with large-scale exports
                               scale = 500,
                               sf = F)


manyroads_light2<- ee_extract(x = NOAA_ligth,
                              y = roads_buffer [1001:1500,],
                              fun = ee$Reducer$mean(),
                              via = "drive", # This will not work with large-scale exports
                              scale = 500,
                              sf = F)

manyroads_light3<- ee_extract(x = NOAA_ligth,
                              y = roads_buffer [1501:2000,],
                              fun = ee$Reducer$mean(),
                              via = "drive", # This will not work with large-scale exports
                              scale = 500,
                              sf = F)

manyroads_light4<- ee_extract(x = NOAA_ligth,
                              y = roads_buffer [2001:2500,],
                              fun = ee$Reducer$mean(),
                              via = "drive", # This will not work with large-scale exports
                              scale = 500,
                              sf = F)

manyroads_light5<- ee_extract(x = NOAA_ligth,
                              y = roads_buffer [2501:2810,],
                              fun = ee$Reducer$mean(),
                              via = "drive", # This will not work with large-scale exports
                              scale = 500,
                              sf = F)


# Переформатировать базу данных

library(stringr)
df_long <- manyroads_light %>% 
        gather(key=Category, value=Measurement, X20170101_cf_cvg:X20201201_cf_cvg, factor_key = TRUE) %>%
        separate(Category, into = c("Category", "Type")) %>% 
        mutate(date = str_extract("X20170101", "[0-9]+")) %>% 
        mutate(Year = substr(date, 1, 4)) %>% 
        mutate (Month = substr(date, 5, 6)) %>% 
        select(-Category, -Type, -date) %>% 
        rename(clouds_cvrg = Measurement)


df_long1 <- manyroads_light1 %>% 
        gather(key=Category, value=Measurement, X20170101_cf_cvg:X20201201_cf_cvg, factor_key = TRUE) %>%
        separate(Category, into = c("Category", "Type")) %>% 
        mutate(date = str_extract("X20170101", "[0-9]+")) %>% 
        mutate(Year = substr(date, 1, 4)) %>% 
        mutate (Month = substr(date, 5, 6)) %>% 
        select(-Category, -Type, -date) %>% 
        rename(clouds_cvrg = Measurement)

df_long2 <- manyroads_light2 %>% 
        gather(key=Category, value=Measurement, X20170101_cf_cvg:X20201201_cf_cvg, factor_key = TRUE) %>%
        separate(Category, into = c("Category", "Type")) %>% 
        mutate(date = str_extract("X20170101", "[0-9]+")) %>% 
        mutate(Year = substr(date, 1, 4)) %>% 
        mutate (Month = substr(date, 5, 6)) %>% 
        select(-Category, -Type, -date) %>% 
        rename(clouds_cvrg = Measurement)

df_long3 <- manyroads_light3 %>% 
        gather(key=Category, value=Measurement, X20170101_cf_cvg:X20201201_cf_cvg, factor_key = TRUE) %>%
        separate(Category, into = c("Category", "Type")) %>% 
        mutate(date = str_extract("X20170101", "[0-9]+")) %>% 
        mutate(Year = substr(date, 1, 4)) %>% 
        mutate (Month = substr(date, 5, 6)) %>% 
        select(-Category, -Type, -date) %>% 
        rename(clouds_cvrg = Measurement)

df_long4 <- manyroads_light4 %>% 
        gather(key=Category, value=Measurement, X20170101_cf_cvg:X20201201_cf_cvg, factor_key = TRUE) %>%
        separate(Category, into = c("Category", "Type")) %>% 
        mutate(date = str_extract("X20170101", "[0-9]+")) %>% 
        mutate(Year = substr(date, 1, 4)) %>% 
        mutate (Month = substr(date, 5, 6)) %>% 
        select(-Category, -Type, -date) %>% 
        rename(clouds_cvrg = Measurement)

df_long5 <- manyroads_light5 %>% 
        gather(key=Category, value=Measurement, X20170101_cf_cvg:X20201201_cf_cvg, factor_key = TRUE) %>%
        separate(Category, into = c("Category", "Type")) %>% 
        mutate(date = str_extract("X20170101", "[0-9]+")) %>% 
        mutate(Year = substr(date, 1, 4)) %>% 
        mutate (Month = substr(date, 5, 6)) %>% 
        select(-Category, -Type, -date) %>% 
        rename(clouds_cvrg = Measurement)

# Объединяем
clouds_coverage <- rbind(df_long, df_long1, df_long2, df_long3, df_long4, df_long5)

write.csv(clouds_coverage, "clouds_coverage.csv")
