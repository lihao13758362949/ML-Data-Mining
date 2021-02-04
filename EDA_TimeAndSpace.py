# 时空特征探索 EDA_TimeAndSpace.py

# 每年工作日取日平均，非工作日取日平均和节假日取日平均，三种情况下出租车&网约车：
# 运营时间规律：出车时间和运行时间；
# 空间分布规律：城市分布规律，订单分布规律；
# 日均空驶率：空驶里程(没有载客)在车辆总运行里程中所占的比例；
# 订单平均运距：订单平均距离计算；
# 订单平均运行时长：订单平时时长计算；
# 上下客点分布密度：上下车位置分布；

# 1 时间数据探索与处理
taxigps2019['GPS_TIME'] = pd.to_datetime(taxigps2019['GPS_TIME'])
taxigps2019['GPS_TIME'].dt.hour.value_counts()

# 2 特征构建
## 统计每辆巡游车的经纬度和速度极差
taxigps2019 = taxigps2019[taxigps2019['LATITUDE'] != 0]
taxigps2019 = taxigps2019[taxigps2019['LONGITUDE'] != 0]

df['LATITUDE_PTP'] = taxigps2019.groupby(['CARNO'])['LATITUDE'].apply(np.ptp)
df['LONGITUDE_PTP'] = taxigps2019.groupby(['CARNO'])['LONGITUDE'].apply(np.ptp)
df['GPS_SPEED_PTP'] = taxigps2019.groupby(['CARNO'])['GPS_SPEED'].apply(np.ptp) #通过统计经纬度是不是全天都为0，我们可以剔除58辆全天GPS都异常的车。





# folium库可视化
## 热度图
from folium import plugins
from folium.plugins import HeatMap

map_hooray = folium.Map(location=[24.482426, 118.157606], zoom_start=14)
HeatMap(taxigps2019[['LATITUDE', 'LONGITUDE']].iloc[:1000].values).add_to(map_hooray)
map_hooray

## 路线图
import folium
# Create the map and add the line
m = folium.Map(location=[24.482426, 118.157606], zoom_start=12)
my_PolyLine=folium.PolyLine(locations=taxigps2019[taxigps2019['CARNO'] == '0006d282be70d06881a7513b69fcaa60'][['LATITUDE', 'LONGITUDE']].iloc[:50].values,weight=5)
m.add_children(my_PolyLine)
