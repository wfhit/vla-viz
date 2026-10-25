import { LayerView } from '@/src/vla/LayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
  title: 'GR00T N1.7 可视化',
  description: 'NVIDIA GR00T N1.7（System 2 视觉语言主干 + System 1 DiT 动作头）的 3D 讲解：真实权重、真实激活。',
};

export default function Page() {
    return <>
        <Header title="GR00T N1.7 可视化" />
        <LayerView pageId="groot" />
        <div id="portal-container"></div>
    </>;
}
