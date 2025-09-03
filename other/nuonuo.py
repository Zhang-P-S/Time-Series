
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView

class NUONUO:
    def __init__(self):
        self.name = "诺诺"
        self.description = "一个可爱甜美的少女"
        self.age = 19
        self.hobby = ["弹琴", "厨艺", "烘焙", "园艺"]
        self.boyfriend = "小章学长"

    def greet(self):
        return f"{self.boyfriend} loves {self.name} !"
    
    def get_info(self):
        hobbies = ", ".join(self.hobby)
        return f"姓名: {self.name}\n年龄: {self.age}\n描述: {self.description}\n爱好: {hobbies}\n男友: {self.boyfriend}"

class NuonuoApp(App):
    def build(self):
        self.nuonuo = NUONUO()
        
        # 设置窗口背景色为粉色
        Window.clearcolor = (1, 0.8, 0.9, 1)
        
        # 创建主布局
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        
        # 添加标题
        title = Label(text="诺诺的信息", font_size=24, size_hint_y=None, height=50)
        layout.add_widget(title)
        
        # 添加图片（使用占位图）
        # 在实际应用中，你可以替换为真实的图片
        image = Image(source='nuonuo.png', size_hint_y=None, height=200)
        layout.add_widget(image)
        
        # 添加信息标签
        info_label = Label(text=self.nuonuo.get_info(), font_size=18, size_hint_y=None, height=200)
        layout.add_widget(info_label)
        
        # 添加问候按钮
        greet_btn = Button(text="打招呼", size_hint_y=None, height=60)
        greet_btn.bind(on_press=self.show_greeting)
        layout.add_widget(greet_btn)
        
        # 添加问候结果显示标签
        self.greeting_label = Label(text="", font_size=20, size_hint_y=None, height=50)
        layout.add_widget(self.greeting_label)
        
        return layout
    
    def show_greeting(self, instance):
        self.greeting_label.text = self.nuonuo.greet()

if __name__ == '__main__':
    NuonuoApp().run()