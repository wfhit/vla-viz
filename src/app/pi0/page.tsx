import { LayerView } from '@/src/vla/LayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
  title: 'π0.5 可视化',
  description: 'π0.5（PaliGemma 前缀 + 动作专家，逐层联合注意力、adaRMSNorm 时间条件、10 步 Euler）的 3D 讲解：真实权重、真实激活。',
};

export default function Page() {
    return <>
        <Header title="π0.5 可视化" />
        <LayerView pageId="pi0" />
        <div id="portal-container"></div>
    </>;
}
