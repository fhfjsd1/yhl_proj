idf_component_register(
    SRCS "blink_example_main.c" "I2S/I2Sto.c"
    INCLUDE_DIRS "." "./I2S"
    # REQUIRES "hardware_driver" "sr_ringbuf" "player"
)

add_definitions(-w)

# 从构建属性中获取所有组件
idf_build_get_property(build_components BUILD_COMPONENTS)  # 获取构建中的所有组件列表

# 判断 esp-sr 是否在组件列表中
if(esp-sr IN_LIST build_components)
    # 如果存在，则使用本地组件名称
    set(esp-sr_name esp-sr)  # 设置 esp-sr_name 为 esp-sr（本地组件）
else()
    # 如果不存在，则使用托管组件名称
    set(esp-sr_name espressif__esp-sr)  # 设置 esp-sr_name 为 espressif__esp-sr（托管组件）
endif()

# 获取 esp-sr 组件所在的目录
idf_component_get_property(esp-sr_path ${esp-sr_name} COMPONENT_DIR)  # 获取 esp-sr 组件目录

# 设置 voice_data_image 路径为项目目录下的语音数据文件
set(voice_data_image D:/ESP_exp/I2S/esp_tts_voice_data_xiaoxin.dat)  # 使用项目目录下的语音数据文件

# 创建 voice_data 自定义目标，确保在构建时生成该文件
add_custom_target(voice_data ALL DEPENDS ${voice_data_image})  # 定义 voice_data 目标依赖语音数据文件

# 添加 flash 目标与 voice_data 目标之间的依赖关系
add_dependencies(flash voice_data)  # 确保 flash 前先构建 voice_data

# 从分区表中获取 voice_data 分区的大小信息
partition_table_get_partition_info(size "--partition-name voice_data" "size")  # 获取 voice_data 分区的大小

# 从分区表中获取 voice_data 分区的偏移信息
partition_table_get_partition_info(offset "--partition-name voice_data" "offset")  # 获取 voice_data 分区的偏移量

# 如果成功获取到分区大小和偏移信息
if("${size}" AND "${offset}")
    # 调用 esptool_py_flash_to_partition 命令将语音数据刷入 voice_data 分区
    esptool_py_flash_to_partition(flash "voice_data" "${voice_data_image}")  # 刷入语音数据分区
else()
    # 如果没有找到分区信息，设置警告信息提醒用户修改分区配置文件
    set(message "Failed to find model in partition table file"
                "Please add a line(Name=voice_data, Type=data, Size=3890K) to the partition file.")  # 设置错误信息
endif()
