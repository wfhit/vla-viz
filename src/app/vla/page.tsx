import { LayerView } from '@/src/vla/LayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
  title: 'k005b VLA 可视化',
  description: 'k005b VLA 网络（Qwen3-VL 主干 + 动作专家）的 3D 动画讲解。',
};

export default function Page() {
    return <>
        <Header title="k005b VLA 可视化" />
        <LayerView />
        <div id="portal-container"></div>
    </>;
}
