-- Active: 1691744613924@@127.0.0.1@3306


USE berlin_accident;
        
-- Accident Data Table
CREATE TABLE accident_data (
  key_id VARCHAR(255) NOT NULL COMMENT 'object_id + year + lor?',
  object_id VARCHAR(255) NOT NULL,
  street_default VARCHAR(255) NULL COMMENT 'old data until 2019',
  lor INT NOT NULL,
  year INT NULL,
  month INT NULL,
  weekday INT NULL COMMENT 'sunday = 1',
  ac_category INT NULL,
  ac_type1 INT NULL,
  ac_type2 INT NULL,
  ac_light INT NULL,
  is_bicycle INT NULL,
  is_car INT NULL,
  is_pedestrian INT NULL,
  is_motorcycle INT NULL,
  is_truck INT NULL,
  is_other INT NULL,
  linrefx FLOAT NULL,
  linrefy FLOAT NULL,
  xgk3 FLOAT NULL COMMENT 'XGCSWGS84',
  ygk3 FLOAT NULL COMMENT 'YGCSWGS84',
  PRIMARY KEY (key_id)
);

-- City Structure Table
CREATE TABLE city_structure (
  structure_type_id VARCHAR(255) NOT NULL,
  structure_type VARCHAR(255) NOT NULL,
  PRIMARY KEY (structure_type_id)
);

-- District Index Table
CREATE TABLE district_index (
  district_id INT NOT NULL,
  district VARCHAR(255) NOT NULL,
  PRIMARY KEY (district_id)
);

-- Land Use Table
CREATE TABLE land_use (
  land_use_id INT NOT NULL,
  land_use VARCHAR(255) NULL,
  PRIMARY KEY (land_use_id)
);

-- LOR 2021 Table
CREATE TABLE LOR_2021 (
  lor INT NOT NULL,
  district INT NOT NULL COMMENT 'LOR - 1',
  forecast_area INT NULL COMMENT 'LOR - 2',
  district_region INT NULL COMMENT 'LOR - 3',
  planning_area INT NULL COMMENT 'LOR - 4',
  sub_district VARCHAR(255) NULL,
  street_code INT NULL,
  street_name VARCHAR(255) NULL,
  street_range VARCHAR(255) NULL COMMENT '??',
  district_id INT NOT NULL,
  PRIMARY KEY (lor)
);

-- Population Data Table
CREATE TABLE population_data (
  key_id VARCHAR(255) NOT NULL COMMENT 'LOR + year + half_year',
  lor INT NOT NULL,
  year INT NULL,
  half_year INT NULL,
  total_population INT NULL,
  age_group_0_6 INT NULL,
  age_group_6_15 INT NULL,
  age_group_15_18 INT NULL,
  age_group_18_27 INT NULL,
  age_group_27_45 INT NULL,
  age_group_45_55 INT NULL,
  age_group_55_65 INT NULL,
  age_group_65_plus INT NULL,
  PRIMARY KEY (key_id)
);

-- Urban Data Table
CREATE TABLE urban_data (
  key_id INT NOT NULL,
  district_id INT NOT NULL,
  structure_type_id INT NOT NULL,
  land_use_id INT NOT NULL,
  area_size INT NULL,
  PRIMARY KEY (key_id)
);

-- Adding Foreign Key Constraints
ALTER TABLE urban_data ADD CONSTRAINT FK_district_index_TO_urban_data FOREIGN KEY (district_id) REFERENCES district_index (district_id);
ALTER TABLE urban_data ADD CONSTRAINT FK_city_structure_TO_urban_data FOREIGN KEY (structure_type_id) REFERENCES city_structure (structure_type_id);
ALTER TABLE urban_data ADD CONSTRAINT FK_land_use_TO_urban_data FOREIGN KEY (land_use_id) REFERENCES land_use (land_use_id);
ALTER TABLE population_data ADD CONSTRAINT FK_LOR_2021_TO_population_data FOREIGN KEY (lor) REFERENCES LOR_2021 (lor);
ALTER TABLE LOR_2021 ADD CONSTRAINT FK_district_index_TO_LOR_2021 FOREIGN KEY (district_id) REFERENCES district_index (district_id);
ALTER TABLE accident_data ADD CONSTRAINT FK_LOR_2021_TO_accident_data FOREIGN KEY (lor) REFERENCES LOR_2021 (lor);
