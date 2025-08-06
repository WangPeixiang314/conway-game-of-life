import pygame
import numpy as np
import numba as nb
import time
import sys

# 超参数设置
# 网格参数
GRID_WIDTH = 800     # 网格宽度（单元格数）
GRID_HEIGHT = 440    # 网格高度（单元格数）
CELL_SIZE = 2        # 单元格像素大小
INITIAL_DENSITY = 0.41  # 初始细胞密度
CLUSTER_RADIUS = 345   # 初始细胞聚集半径
CLUSTER_COUNT = 4  # 聚集中心数量

# UI参数
UI_HEIGHT = 100      # UI区域高度
BUTTON_WIDTH = 80    # 标准按钮宽度
BUTTON_HEIGHT = 30   # 按钮高度
BUTTON_SPACING = 10  # 按钮间距
SMALL_BUTTON_WIDTH = 40  # 小按钮宽度
max_brush_size = 30  # 最大笔刷大小

# 帧率控制
MIN_FPS = 15         # 最小帧率
MAX_FPS = 300        # 最大帧率
FPS_STEP = 15        # 帧率调节步长
DEFAULT_FPS = 60     # 默认帧率

# 颜色设置
BACKGROUND_COLOR = (15, 20, 30)
GRID_COLOR = (30, 35, 45)
CELL_COLOR = (97, 175, 239)
TEXT_COLOR = (220, 220, 220)
HIGHLIGHT_COLOR = (255, 215, 0)
UI_BG_COLOR = (30, 35, 45, 200)
BUTTON_COLOR = (50, 60, 80)
BUTTON_HOVER_COLOR = (70, 90, 110)
BUTTON_ACTIVE_COLOR = (97, 175, 239)
BUTTON_DISABLED_COLOR = (30, 35, 45)

# 计算窗口尺寸
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE + UI_HEIGHT

# 初始化Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("康威生命游戏 - GPU/CPU 版")
clock = pygame.time.Clock()

# 创建UI区域
ui_surface = pygame.Surface((WINDOW_WIDTH, UI_HEIGHT), pygame.SRCALPHA)
ui_surface.fill(UI_BG_COLOR)

# 设置字体
try:
    # 尝试使用系统自带的中文字体
    font = pygame.font.SysFont('SimHei', 18)
    title_font = pygame.font.SysFont('SimHei', 24, bold=True)
    button_font = pygame.font.SysFont('SimHei', 16)
    # 测试字体是否能渲染中文
    test_text = font.render("测试", True, (255, 255, 255))
    if test_text.get_size()[0] == 0:
        font = pygame.font.SysFont('Microsoft YaHei', 18)
        title_font = pygame.font.SysFont('Microsoft YaHei', 24, bold=True)
        button_font = pygame.font.SysFont('Microsoft YaHei', 16)
except:
    try:
        # 尝试加载本地字体文件
        font = pygame.font.Font("simhei.ttf", 18)
        title_font = pygame.font.Font("simhei.ttf", 24)
        button_font = pygame.font.Font("simhei.ttf", 16)
    except:
        # 最后回退方案
        font = pygame.font.SysFont(None, 18)
        title_font = pygame.font.SysFont(None, 24)
        button_font = pygame.font.SysFont(None, 16)
        print("警告: 未找到中文字体，中文显示可能不正常")

# 检查CUDA是否可用
GPU_AVAILABLE = False
try:
    from numba import cuda
    if cuda.is_available():
        print("检测到CUDA设备，GPU加速可用")
        GPU_AVAILABLE = True
    else:
        print("未检测到CUDA设备，GPU加速不可用")
except ImportError:
    print("无法导入numba.cuda，GPU加速不可用")

# 初始设备选择
USE_GPU = GPU_AVAILABLE  # 默认使用GPU（如果可用）

# GPU内核函数（如果可用）
if GPU_AVAILABLE:
    @cuda.jit
    def update_grid_gpu(grid, new_grid):
        i, j = cuda.grid(2)
        rows, cols = grid.shape
        
        if i < rows and j < cols:
            # 计算周围活细胞数量（使用周期边界）
            neighbors = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    
                    ni = (i + di) % rows
                    nj = (j + dj) % cols
                    neighbors += grid[ni, nj]
            
            # 应用生命游戏规则
            if grid[i, j] == 1:
                if neighbors == 2 or neighbors == 3:
                    new_grid[i, j] = 1
                else:
                    new_grid[i, j] = 0
            else:
                if neighbors == 3:
                    new_grid[i, j] = 1
                else:
                    new_grid[i, j] = 0

# 优化后的CPU函数
@nb.njit(nb.uint8[:,:](nb.uint8[:,:]), cache=True, parallel=True, fastmath=True)
def update_grid_cpu(grid):
    new_grid = np.zeros_like(grid)
    rows, cols = grid.shape
    
    for i in nb.prange(rows):
        for j in nb.prange(cols):
            # 计算周围活细胞数量（使用周期边界）
            neighbors = 0
            # 优化：减少边界检查
            i_up = (i - 1) % rows
            i_down = (i + 1) % rows
            j_left = (j - 1) % cols
            j_right = (j + 1) % cols
            
            neighbors = (
                grid[i_up, j_left] + grid[i_up, j] + grid[i_up, j_right] +
                grid[i, j_left] + grid[i, j_right] +
                grid[i_down, j_left] + grid[i_down, j] + grid[i_down, j_right]
            )
            
            # 应用生命游戏规则
            if grid[i, j] == 1:
                if neighbors == 2 or neighbors == 3:
                    new_grid[i, j] = 1
            else:
                if neighbors == 3:
                    new_grid[i, j] = 1
                    
    return new_grid

# 创建初始网格（带有聚集效果的细胞分布）
@nb.njit(cache=True, fastmath=True)
def create_initial_grid():
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
    total_cells = GRID_WIDTH * GRID_HEIGHT
    num_alive = int(total_cells * INITIAL_DENSITY)
    
    # 创建几个聚集中心
    cluster_centers = np.array([
        (np.random.randint(GRID_HEIGHT//4, 3*GRID_HEIGHT//4), 
        np.random.randint(GRID_WIDTH//4, 3*GRID_WIDTH//4))
        for _ in range(CLUSTER_COUNT)
    ])
    
    # 在聚集中心周围生成细胞
    placed = 0
    max_attempts = num_alive * 10  # 最大尝试次数，避免无限循环
    attempts = 0
    
    while placed < num_alive and attempts < max_attempts:
        # 随机选择一个聚集中心
        idx = np.random.choice(len(cluster_centers))
        cx, cy = cluster_centers[idx]
        
        # 在中心点附近随机偏移（使用高斯分布）
        dx = int(np.random.normal(0, CLUSTER_RADIUS))
        dy = int(np.random.normal(0, CLUSTER_RADIUS))
        x = (cx + dx) % GRID_HEIGHT
        y = (cy + dy) % GRID_WIDTH
        
        # 如果位置为空则放置细胞
        if grid[x, y] == 0:
            grid[x, y] = 1
            placed += 1
        
        attempts += 1
    
    # 如果未能放置足够的细胞，使用随机填充补足
    if placed < num_alive:
        remaining = num_alive - placed
        indices = np.random.choice(total_cells, remaining, replace=False)
        for idx in indices:
            x = idx // GRID_WIDTH
            y = idx % GRID_WIDTH
            grid[x, y] = 1
    
    return grid

# 按钮类
class Button:
    def __init__(self, x, y, width, height, text, action=None, enabled=True):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.hovered = False
        self.active = False
        self.was_clicked = False  # 跟踪按钮是否被点击
        self.enabled = enabled    # 按钮是否可用
        
    def draw(self, surface):
        # 根据状态选择颜色
        if not self.enabled:
            color = BUTTON_DISABLED_COLOR
        elif self.was_clicked:
            color = BUTTON_ACTIVE_COLOR
            self.was_clicked = False  # 重置点击状态
        elif self.hovered:
            color = BUTTON_HOVER_COLOR
        else:
            color = BUTTON_COLOR
            
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        
        if self.enabled:
            pygame.draw.rect(surface, (100, 120, 140), self.rect, 2, border_radius=4)
        
        text_color = TEXT_COLOR if self.enabled else (100, 100, 100)
        text_surf = button_font.render(self.text, True, text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
        
    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos) and self.enabled
        return self.hovered
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered and self.enabled:
                self.was_clicked = True  # 标记按钮被点击
                return True
        return False

# 绘制UI元素
def draw_ui(buttons, device, compute_time, generation, alive_count, paused, brush_size, show_grid, save_mode, fps):
    ui_surface.fill(UI_BG_COLOR)
    
    # 标题
    title = title_font.render("康威生命游戏 - GPU/CPU 版", True, HIGHLIGHT_COLOR)
    ui_surface.blit(title, (10, 5))

    # 合并其他信息到一个f-string
    info_text = f"设备: {device} | 计算时间: {compute_time:.3f}ms | 帧率: {fps} FPS | 代数: {generation} | 活细胞: {alive_count} ({alive_count/(GRID_WIDTH*GRID_HEIGHT)*100:.1f}%) | 状态: {'已暂停' if paused else '运行中'} | 保存模式: {'开' if save_mode else '关'} | 画笔大小: {brush_size}"
    info_surface = font.render(info_text, True, TEXT_COLOR)
    ui_surface.blit(info_surface, (25, 35))
    
    # 绘制按钮
    for button in buttons:
        button.draw(ui_surface)
    
    screen.blit(ui_surface, (0, GRID_HEIGHT * CELL_SIZE))

# 主游戏循环
def main():
    global USE_GPU
    
    grid = create_initial_grid()
    running = True
    paused = False
    generation = 0
    brush_size = 3
    show_grid = True
    last_update_time = time.time()
    compute_time = 0
    save_mode = False
    fps = DEFAULT_FPS  # 初始帧率
    
    # CUDA相关变量
    d_grid = None
    d_new_grid = None
    blocks_per_grid = None
    threads_per_block = None
    
    # 初始化CUDA（如果GPU可用且被选中）
    def init_cuda():
        nonlocal d_grid, d_new_grid, blocks_per_grid, threads_per_block
        
        if GPU_AVAILABLE and USE_GPU:
            # 设置CUDA线程块和网格大小
            threads_per_block = (32, 32)
            blocks_per_grid_x = (GRID_HEIGHT + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (GRID_WIDTH + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            
            # 创建设备数组
            d_grid = cuda.to_device(grid)
            d_new_grid = cuda.device_array_like(grid)
            print("已初始化CUDA设备")
        else:
            d_grid = None
            d_new_grid = None
            blocks_per_grid = None
            threads_per_block = None
    
    # 初始化CUDA
    init_cuda()
    
   # 创建按钮
    buttons = [
        Button(20, 60, 120, 30, "暂停/继续", "pause"),
        Button(160, 60, 120, 30, "重置", "reset"),
        Button(300, 60, 120, 30, "切换网格", "grid"),
        Button(440, 60, 120, 30, "清空", "clear"),
        Button(580, 60, 120, 30, "保存模式", "save"),
        Button(720, 60, 120, 30, "减小画笔", "brush_down"),
        Button(860, 60, 120, 30, "增大画笔", "brush_up"),
        # 新增帧率控制按钮
        Button(1000, 60, 120, 30, "帧率-", "fps_down"),
        Button(1140, 60, 120, 30, "帧率+", "fps_up"),
        # 新增设备切换按钮
        Button(1280, 60, 120, 30, "切换到CPU", "switch_device", enabled=GPU_AVAILABLE),
    ]
    
    # 保存设备切换按钮的引用
    device_button = buttons[-1]
    
    while running:
        current_time = time.time()
        mouse_pos = pygame.mouse.get_pos()
        mouse_grid_pos = (mouse_pos[0] // CELL_SIZE, mouse_pos[1] // CELL_SIZE)
        
        # 更新设备按钮文本
        if GPU_AVAILABLE:
            device_button.text = "切换到CPU" if USE_GPU else "切换到GPU"
        
        # UI区域鼠标位置（用于按钮）
        ui_mouse_pos = (mouse_pos[0], mouse_pos[1] - GRID_HEIGHT * CELL_SIZE) if mouse_pos[1] > GRID_HEIGHT * CELL_SIZE else None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # 按钮事件处理
            if ui_mouse_pos:
                for button in buttons:
                    button.check_hover(ui_mouse_pos)
                    if button.handle_event(event):
                        if button.action == "pause":
                            paused = not paused
                        elif button.action == "reset":
                            grid = create_initial_grid()
                            generation = 0
                            paused = False
                            if GPU_AVAILABLE and USE_GPU:
                                d_grid = cuda.to_device(grid)
                        elif button.action == "grid":
                            show_grid = not show_grid
                        elif button.action == "clear":
                            grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
                            generation = 0
                            if GPU_AVAILABLE and USE_GPU:
                                d_grid = cuda.to_device(grid)
                        elif button.action == "save":
                            save_mode = not save_mode
                        elif button.action == "brush_down":
                            brush_size = max(brush_size - 1, 1)
                        elif button.action == "brush_up":
                            brush_size = min(brush_size + 1, max_brush_size)
                        elif button.action == "fps_down":
                            fps = max(fps - FPS_STEP, MIN_FPS)
                        elif button.action == "fps_up":
                            fps = min(fps + FPS_STEP, MAX_FPS)
                        elif button.action == "switch_device" and GPU_AVAILABLE:
                            # 切换设备
                            USE_GPU = not USE_GPU
                            # 重新初始化CUDA
                            init_cuda()
            
            # 鼠标点击切换细胞状态
            elif event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]):
                if mouse_pos[1] < GRID_HEIGHT * CELL_SIZE:  # 确保在网格区域内
                    grid_x, grid_y = mouse_grid_pos
                    
                    # 使用画笔大小
                    for dx in range(-brush_size + 1, brush_size):
                        for dy in range(-brush_size + 1, brush_size):
                            nx = grid_y + dx
                            ny = grid_x + dy
                            
                            # 确保在网格范围内
                            if 0 <= nx < GRID_HEIGHT and 0 <= ny < GRID_WIDTH:
                                # 圆形画笔
                                if dx*dx + dy*dy < brush_size*brush_size:
                                    if save_mode:
                                        grid[nx, ny] = 1
                                    else:
                                        grid[nx, ny] = 1 - grid[nx, ny]
                    
                    if GPU_AVAILABLE and USE_GPU:
                        d_grid = cuda.to_device(grid)
        
        # 更新游戏状态
        if not paused:
            start_time = time.perf_counter()
            
            if GPU_AVAILABLE and USE_GPU and d_grid is not None:
                # GPU版本
                update_grid_gpu[blocks_per_grid, threads_per_block](d_grid, d_new_grid)
                d_grid.copy_to_host(grid)
                d_grid, d_new_grid = d_new_grid, d_grid
                device = "GPU"
            else:
                # CPU版本
                grid = update_grid_cpu(grid)
                device = "CPU"
            
            compute_time = (time.perf_counter() - start_time) * 1000  # 毫秒
            generation += 1  # 增加代数计数
        
        # 渲染网格
        screen.fill(BACKGROUND_COLOR)
        
        # 绘制单元格
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                if grid[i, j] == 1:
                    pygame.draw.rect(
                        screen, 
                        CELL_COLOR, 
                        (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    )
        
        # 绘制网格线
        if show_grid:
            # 水平线
            for i in range(0, GRID_HEIGHT + 1, 5):  # 每5格画一条线
                pygame.draw.line(
                    screen, 
                    GRID_COLOR, 
                    (0, i * CELL_SIZE), 
                    (WINDOW_WIDTH, i * CELL_SIZE),
                    1
                )
            # 垂直线
            for j in range(0, GRID_WIDTH + 1, 5):  # 每5格画一条线
                pygame.draw.line(
                    screen, 
                    GRID_COLOR, 
                    (j * CELL_SIZE, 0), 
                    (j * CELL_SIZE, GRID_HEIGHT * CELL_SIZE),
                    1
                )
        
        # 绘制鼠标位置指示器
        if mouse_pos[1] < GRID_HEIGHT * CELL_SIZE:
            grid_x, grid_y = mouse_grid_pos
            pygame.draw.rect(
                screen,
                HIGHLIGHT_COLOR,
                (grid_x * CELL_SIZE, grid_y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                1
            )
            
            # 绘制画笔范围
            pygame.draw.circle(
                screen,
                (255, 215, 0, 100),
                (mouse_pos[0], mouse_pos[1]),
                brush_size * CELL_SIZE,
                1
            )
        
        # 显示UI - 包含代数信息
        alive_count = np.sum(grid)
        draw_ui(buttons, device, compute_time, generation, alive_count, paused, brush_size, show_grid, save_mode, fps)
        
        pygame.display.flip()
        clock.tick(fps)  # 使用可调节的帧率
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()