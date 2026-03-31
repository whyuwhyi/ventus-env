; ModuleID = 'kernel.cl'
source_filename = "kernel.cl"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128-A5-G1"
target triple = "riscv32"

; Function Attrs: convergent nofree norecurse nounwind memory(argmem: readwrite) vscale_range(1,2048)
define dso_local ventus_kernel void @gelu_mlp_fp16(ptr addrspace(1) nocapture noundef readonly align 4 %x_tiles, ptr addrspace(1) nocapture noundef readonly align 4 %w1_tiles, ptr addrspace(1) nocapture noundef readonly align 4 %w2_tiles, ptr addrspace(1) nocapture noundef writeonly align 4 %out_tiles) local_unnamed_addr #0 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
entry:
  %call = call i32 @_Z13get_global_idj(i32 noundef 0) #4
  br label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %entry, %for.cond.cleanup7
  %outputs.sroa.22.0 = phi <8 x float> [ zeroinitializer, %entry ], [ %53, %for.cond.cleanup7 ]
  %outputs.sroa.18.0 = phi <8 x float> [ zeroinitializer, %entry ], [ %48, %for.cond.cleanup7 ]
  %outputs.sroa.14.0 = phi <8 x float> [ zeroinitializer, %entry ], [ %43, %for.cond.cleanup7 ]
  %outputs.sroa.10.0 = phi <8 x float> [ zeroinitializer, %entry ], [ %38, %for.cond.cleanup7 ]
  %outputs.sroa.6.0 = phi <8 x float> [ zeroinitializer, %entry ], [ %33, %for.cond.cleanup7 ]
  %outputs.sroa.0.0 = phi <8 x float> [ zeroinitializer, %entry ], [ %28, %for.cond.cleanup7 ]
  %h.0105 = phi i32 [ 0, %entry ], [ %inc31, %for.cond.cleanup7 ]
  %mul = mul nuw nsw i32 %h.0105, 6
  br label %for.body8

for.cond.cleanup7:                                ; preds = %for.body8
  %0 = extractelement <8 x float> %62, i64 0
  %1 = call float @llvm.riscv.ventus.vgelu.approx.f32(float %0)
  %2 = extractelement <8 x float> %62, i64 1
  %3 = call float @llvm.riscv.ventus.vgelu.approx.f32(float %2)
  %4 = extractelement <8 x float> %62, i64 2
  %5 = call float @llvm.riscv.ventus.vgelu.approx.f32(float %4)
  %6 = extractelement <8 x float> %62, i64 3
  %7 = call float @llvm.riscv.ventus.vgelu.approx.f32(float %6)
  %8 = extractelement <8 x float> %62, i64 4
  %9 = call float @llvm.riscv.ventus.vgelu.approx.f32(float %8)
  %10 = extractelement <8 x float> %62, i64 5
  %11 = call float @llvm.riscv.ventus.vgelu.approx.f32(float %10)
  %12 = extractelement <8 x float> %62, i64 6
  %13 = call float @llvm.riscv.ventus.vgelu.approx.f32(float %12)
  %14 = extractelement <8 x float> %62, i64 7
  %15 = call float @llvm.riscv.ventus.vgelu.approx.f32(float %14)
  %16 = call i32 @llvm.riscv.ventus.vcvt.fp16.fp32(float %1)
  %and.i = and i32 %16, 65535
  %17 = call i32 @llvm.riscv.ventus.vcvt.fp16.fp32(float %3)
  %18 = call i32 @llvm.riscv.ventus.vcvt.fp16.fp32(float %5)
  %and2.i = and i32 %18, 65535
  %19 = call i32 @llvm.riscv.ventus.vcvt.fp16.fp32(float %7)
  %20 = call i32 @llvm.riscv.ventus.vcvt.fp16.fp32(float %9)
  %and4.i = and i32 %20, 65535
  %21 = call i32 @llvm.riscv.ventus.vcvt.fp16.fp32(float %11)
  %22 = call i32 @llvm.riscv.ventus.vcvt.fp16.fp32(float %13)
  %and6.i = and i32 %22, 65535
  %23 = call i32 @llvm.riscv.ventus.vcvt.fp16.fp32(float %15)
  %shl.i = shl i32 %17, 16
  %or.i = or i32 %shl.i, %and.i
  %vecinit.i67 = insertelement <4 x i32> undef, i32 %or.i, i64 0
  %shl8.i = shl i32 %19, 16
  %or9.i = or i32 %shl8.i, %and2.i
  %vecinit10.i = insertelement <4 x i32> %vecinit.i67, i32 %or9.i, i64 1
  %shl11.i = shl i32 %21, 16
  %or12.i = or i32 %shl11.i, %and4.i
  %vecinit13.i = insertelement <4 x i32> %vecinit10.i, i32 %or12.i, i64 2
  %shl14.i = shl i32 %23, 16
  %or15.i = or i32 %shl14.i, %and6.i
  %vecinit16.i = insertelement <4 x i32> %vecinit13.i, i32 %or15.i, i64 3
  %mul.i84 = mul i32 %h.0105, 192
  %add.i85 = add nsw i32 %mul.i84, %call
  %mul1.i86 = shl nsw i32 %add.i85, 2
  %arrayidx.i87 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %mul1.i86
  %24 = load i32, ptr addrspace(1) %arrayidx.i87, align 4, !tbaa !10
  %vecinit.i88 = insertelement <4 x i32> undef, i32 %24, i64 0
  %add3.i89 = or i32 %mul1.i86, 1
  %arrayidx4.i90 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add3.i89
  %25 = load i32, ptr addrspace(1) %arrayidx4.i90, align 4, !tbaa !10
  %vecinit5.i91 = insertelement <4 x i32> %vecinit.i88, i32 %25, i64 1
  %add6.i92 = or i32 %mul1.i86, 2
  %arrayidx7.i93 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add6.i92
  %26 = load i32, ptr addrspace(1) %arrayidx7.i93, align 4, !tbaa !10
  %vecinit8.i94 = insertelement <4 x i32> %vecinit5.i91, i32 %26, i64 2
  %add9.i95 = or i32 %mul1.i86, 3
  %arrayidx10.i96 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add9.i95
  %27 = load i32, ptr addrspace(1) %arrayidx10.i96, align 4, !tbaa !10
  %vecinit11.i97 = insertelement <4 x i32> %vecinit8.i94, i32 %27, i64 3
  %28 = call <8 x float> @llvm.riscv.ventus.mma.m16n16k16.row.col.f32.f16.f16.f32(<4 x i32> %vecinit16.i, <4 x i32> %vecinit11.i97, <8 x float> %outputs.sroa.0.0)
  %add22.1 = mul i32 %h.0105, 192
  %mul.i84.1 = or i32 %add22.1, 32
  %add.i85.1 = add nsw i32 %mul.i84.1, %call
  %mul1.i86.1 = shl nsw i32 %add.i85.1, 2
  %arrayidx.i87.1 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %mul1.i86.1
  %29 = load i32, ptr addrspace(1) %arrayidx.i87.1, align 4, !tbaa !10
  %vecinit.i88.1 = insertelement <4 x i32> undef, i32 %29, i64 0
  %add3.i89.1 = or i32 %mul1.i86.1, 1
  %arrayidx4.i90.1 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add3.i89.1
  %30 = load i32, ptr addrspace(1) %arrayidx4.i90.1, align 4, !tbaa !10
  %vecinit5.i91.1 = insertelement <4 x i32> %vecinit.i88.1, i32 %30, i64 1
  %add6.i92.1 = or i32 %mul1.i86.1, 2
  %arrayidx7.i93.1 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add6.i92.1
  %31 = load i32, ptr addrspace(1) %arrayidx7.i93.1, align 4, !tbaa !10
  %vecinit8.i94.1 = insertelement <4 x i32> %vecinit5.i91.1, i32 %31, i64 2
  %add9.i95.1 = or i32 %mul1.i86.1, 3
  %arrayidx10.i96.1 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add9.i95.1
  %32 = load i32, ptr addrspace(1) %arrayidx10.i96.1, align 4, !tbaa !10
  %vecinit11.i97.1 = insertelement <4 x i32> %vecinit8.i94.1, i32 %32, i64 3
  %33 = call <8 x float> @llvm.riscv.ventus.mma.m16n16k16.row.col.f32.f16.f16.f32(<4 x i32> %vecinit16.i, <4 x i32> %vecinit11.i97.1, <8 x float> %outputs.sroa.6.0)
  %add22.2 = mul i32 %h.0105, 192
  %mul.i84.2 = add i32 %add22.2, 64
  %add.i85.2 = add nsw i32 %mul.i84.2, %call
  %mul1.i86.2 = shl nsw i32 %add.i85.2, 2
  %arrayidx.i87.2 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %mul1.i86.2
  %34 = load i32, ptr addrspace(1) %arrayidx.i87.2, align 4, !tbaa !10
  %vecinit.i88.2 = insertelement <4 x i32> undef, i32 %34, i64 0
  %add3.i89.2 = or i32 %mul1.i86.2, 1
  %arrayidx4.i90.2 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add3.i89.2
  %35 = load i32, ptr addrspace(1) %arrayidx4.i90.2, align 4, !tbaa !10
  %vecinit5.i91.2 = insertelement <4 x i32> %vecinit.i88.2, i32 %35, i64 1
  %add6.i92.2 = or i32 %mul1.i86.2, 2
  %arrayidx7.i93.2 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add6.i92.2
  %36 = load i32, ptr addrspace(1) %arrayidx7.i93.2, align 4, !tbaa !10
  %vecinit8.i94.2 = insertelement <4 x i32> %vecinit5.i91.2, i32 %36, i64 2
  %add9.i95.2 = or i32 %mul1.i86.2, 3
  %arrayidx10.i96.2 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add9.i95.2
  %37 = load i32, ptr addrspace(1) %arrayidx10.i96.2, align 4, !tbaa !10
  %vecinit11.i97.2 = insertelement <4 x i32> %vecinit8.i94.2, i32 %37, i64 3
  %38 = call <8 x float> @llvm.riscv.ventus.mma.m16n16k16.row.col.f32.f16.f16.f32(<4 x i32> %vecinit16.i, <4 x i32> %vecinit11.i97.2, <8 x float> %outputs.sroa.10.0)
  %add22.3 = mul i32 %h.0105, 192
  %mul.i84.3 = add i32 %add22.3, 96
  %add.i85.3 = add nsw i32 %mul.i84.3, %call
  %mul1.i86.3 = shl nsw i32 %add.i85.3, 2
  %arrayidx.i87.3 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %mul1.i86.3
  %39 = load i32, ptr addrspace(1) %arrayidx.i87.3, align 4, !tbaa !10
  %vecinit.i88.3 = insertelement <4 x i32> undef, i32 %39, i64 0
  %add3.i89.3 = or i32 %mul1.i86.3, 1
  %arrayidx4.i90.3 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add3.i89.3
  %40 = load i32, ptr addrspace(1) %arrayidx4.i90.3, align 4, !tbaa !10
  %vecinit5.i91.3 = insertelement <4 x i32> %vecinit.i88.3, i32 %40, i64 1
  %add6.i92.3 = or i32 %mul1.i86.3, 2
  %arrayidx7.i93.3 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add6.i92.3
  %41 = load i32, ptr addrspace(1) %arrayidx7.i93.3, align 4, !tbaa !10
  %vecinit8.i94.3 = insertelement <4 x i32> %vecinit5.i91.3, i32 %41, i64 2
  %add9.i95.3 = or i32 %mul1.i86.3, 3
  %arrayidx10.i96.3 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add9.i95.3
  %42 = load i32, ptr addrspace(1) %arrayidx10.i96.3, align 4, !tbaa !10
  %vecinit11.i97.3 = insertelement <4 x i32> %vecinit8.i94.3, i32 %42, i64 3
  %43 = call <8 x float> @llvm.riscv.ventus.mma.m16n16k16.row.col.f32.f16.f16.f32(<4 x i32> %vecinit16.i, <4 x i32> %vecinit11.i97.3, <8 x float> %outputs.sroa.14.0)
  %add22.4 = mul i32 %h.0105, 192
  %mul.i84.4 = add i32 %add22.4, 128
  %add.i85.4 = add nsw i32 %mul.i84.4, %call
  %mul1.i86.4 = shl nsw i32 %add.i85.4, 2
  %arrayidx.i87.4 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %mul1.i86.4
  %44 = load i32, ptr addrspace(1) %arrayidx.i87.4, align 4, !tbaa !10
  %vecinit.i88.4 = insertelement <4 x i32> undef, i32 %44, i64 0
  %add3.i89.4 = or i32 %mul1.i86.4, 1
  %arrayidx4.i90.4 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add3.i89.4
  %45 = load i32, ptr addrspace(1) %arrayidx4.i90.4, align 4, !tbaa !10
  %vecinit5.i91.4 = insertelement <4 x i32> %vecinit.i88.4, i32 %45, i64 1
  %add6.i92.4 = or i32 %mul1.i86.4, 2
  %arrayidx7.i93.4 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add6.i92.4
  %46 = load i32, ptr addrspace(1) %arrayidx7.i93.4, align 4, !tbaa !10
  %vecinit8.i94.4 = insertelement <4 x i32> %vecinit5.i91.4, i32 %46, i64 2
  %add9.i95.4 = or i32 %mul1.i86.4, 3
  %arrayidx10.i96.4 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add9.i95.4
  %47 = load i32, ptr addrspace(1) %arrayidx10.i96.4, align 4, !tbaa !10
  %vecinit11.i97.4 = insertelement <4 x i32> %vecinit8.i94.4, i32 %47, i64 3
  %48 = call <8 x float> @llvm.riscv.ventus.mma.m16n16k16.row.col.f32.f16.f16.f32(<4 x i32> %vecinit16.i, <4 x i32> %vecinit11.i97.4, <8 x float> %outputs.sroa.18.0)
  %add22.5 = mul i32 %h.0105, 192
  %mul.i84.5 = add i32 %add22.5, 160
  %add.i85.5 = add nsw i32 %mul.i84.5, %call
  %mul1.i86.5 = shl nsw i32 %add.i85.5, 2
  %arrayidx.i87.5 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %mul1.i86.5
  %49 = load i32, ptr addrspace(1) %arrayidx.i87.5, align 4, !tbaa !10
  %vecinit.i88.5 = insertelement <4 x i32> undef, i32 %49, i64 0
  %add3.i89.5 = or i32 %mul1.i86.5, 1
  %arrayidx4.i90.5 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add3.i89.5
  %50 = load i32, ptr addrspace(1) %arrayidx4.i90.5, align 4, !tbaa !10
  %vecinit5.i91.5 = insertelement <4 x i32> %vecinit.i88.5, i32 %50, i64 1
  %add6.i92.5 = or i32 %mul1.i86.5, 2
  %arrayidx7.i93.5 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add6.i92.5
  %51 = load i32, ptr addrspace(1) %arrayidx7.i93.5, align 4, !tbaa !10
  %vecinit8.i94.5 = insertelement <4 x i32> %vecinit5.i91.5, i32 %51, i64 2
  %add9.i95.5 = or i32 %mul1.i86.5, 3
  %arrayidx10.i96.5 = getelementptr inbounds i32, ptr addrspace(1) %w2_tiles, i32 %add9.i95.5
  %52 = load i32, ptr addrspace(1) %arrayidx10.i96.5, align 4, !tbaa !10
  %vecinit11.i97.5 = insertelement <4 x i32> %vecinit8.i94.5, i32 %52, i64 3
  %53 = call <8 x float> @llvm.riscv.ventus.mma.m16n16k16.row.col.f32.f16.f16.f32(<4 x i32> %vecinit16.i, <4 x i32> %vecinit11.i97.5, <8 x float> %outputs.sroa.22.0)
  %inc31 = add nuw nsw i32 %h.0105, 1
  %exitcond108.not = icmp eq i32 %inc31, 24
  br i1 %exitcond108.not, label %for.body37.preheader, label %for.cond5.preheader

for.body8:                                        ; preds = %for.cond5.preheader, %for.body8
  %hidden.0103 = phi <8 x float> [ zeroinitializer, %for.cond5.preheader ], [ %62, %for.body8 ]
  %k.0102 = phi i32 [ 0, %for.cond5.preheader ], [ %inc12, %for.body8 ]
  %mul.i = shl nuw nsw i32 %k.0102, 5
  %add.i = add nsw i32 %mul.i, %call
  %mul1.i = shl nsw i32 %add.i, 2
  %arrayidx.i = getelementptr inbounds i32, ptr addrspace(1) %x_tiles, i32 %mul1.i
  %54 = load i32, ptr addrspace(1) %arrayidx.i, align 4, !tbaa !10
  %vecinit.i68 = insertelement <4 x i32> undef, i32 %54, i64 0
  %add3.i = or i32 %mul1.i, 1
  %arrayidx4.i = getelementptr inbounds i32, ptr addrspace(1) %x_tiles, i32 %add3.i
  %55 = load i32, ptr addrspace(1) %arrayidx4.i, align 4, !tbaa !10
  %vecinit5.i69 = insertelement <4 x i32> %vecinit.i68, i32 %55, i64 1
  %add6.i = or i32 %mul1.i, 2
  %arrayidx7.i = getelementptr inbounds i32, ptr addrspace(1) %x_tiles, i32 %add6.i
  %56 = load i32, ptr addrspace(1) %arrayidx7.i, align 4, !tbaa !10
  %vecinit8.i = insertelement <4 x i32> %vecinit5.i69, i32 %56, i64 2
  %add9.i = or i32 %mul1.i, 3
  %arrayidx10.i = getelementptr inbounds i32, ptr addrspace(1) %x_tiles, i32 %add9.i
  %57 = load i32, ptr addrspace(1) %arrayidx10.i, align 4, !tbaa !10
  %vecinit11.i = insertelement <4 x i32> %vecinit8.i, i32 %57, i64 3
  %add = add nuw nsw i32 %k.0102, %mul
  %mul.i70 = shl nsw i32 %add, 5
  %add.i71 = add nsw i32 %mul.i70, %call
  %mul1.i72 = shl nsw i32 %add.i71, 2
  %arrayidx.i73 = getelementptr inbounds i32, ptr addrspace(1) %w1_tiles, i32 %mul1.i72
  %58 = load i32, ptr addrspace(1) %arrayidx.i73, align 4, !tbaa !10
  %vecinit.i74 = insertelement <4 x i32> undef, i32 %58, i64 0
  %add3.i75 = or i32 %mul1.i72, 1
  %arrayidx4.i76 = getelementptr inbounds i32, ptr addrspace(1) %w1_tiles, i32 %add3.i75
  %59 = load i32, ptr addrspace(1) %arrayidx4.i76, align 4, !tbaa !10
  %vecinit5.i77 = insertelement <4 x i32> %vecinit.i74, i32 %59, i64 1
  %add6.i78 = or i32 %mul1.i72, 2
  %arrayidx7.i79 = getelementptr inbounds i32, ptr addrspace(1) %w1_tiles, i32 %add6.i78
  %60 = load i32, ptr addrspace(1) %arrayidx7.i79, align 4, !tbaa !10
  %vecinit8.i80 = insertelement <4 x i32> %vecinit5.i77, i32 %60, i64 2
  %add9.i81 = or i32 %mul1.i72, 3
  %arrayidx10.i82 = getelementptr inbounds i32, ptr addrspace(1) %w1_tiles, i32 %add9.i81
  %61 = load i32, ptr addrspace(1) %arrayidx10.i82, align 4, !tbaa !10
  %vecinit11.i83 = insertelement <4 x i32> %vecinit8.i80, i32 %61, i64 3
  %62 = call <8 x float> @llvm.riscv.ventus.mma.m16n16k16.row.col.f32.f16.f16.f32(<4 x i32> %vecinit11.i, <4 x i32> %vecinit11.i83, <8 x float> %hidden.0103)
  %inc12 = add nuw nsw i32 %k.0102, 1
  %exitcond.not = icmp eq i32 %inc12, 6
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8

for.body37.preheader:                             ; preds = %for.cond.cleanup7
  %arrayidx.i100 = getelementptr inbounds <8 x float>, ptr addrspace(1) %out_tiles, i32 %call
  store <8 x float> %28, ptr addrspace(1) %arrayidx.i100, align 32, !tbaa !14
  %add.i99.1 = add nsw i32 %call, 32
  %arrayidx.i100.1 = getelementptr inbounds <8 x float>, ptr addrspace(1) %out_tiles, i32 %add.i99.1
  store <8 x float> %33, ptr addrspace(1) %arrayidx.i100.1, align 32, !tbaa !14
  %add.i99.2 = add nsw i32 %call, 64
  %arrayidx.i100.2 = getelementptr inbounds <8 x float>, ptr addrspace(1) %out_tiles, i32 %add.i99.2
  store <8 x float> %38, ptr addrspace(1) %arrayidx.i100.2, align 32, !tbaa !14
  %add.i99.3 = add nsw i32 %call, 96
  %arrayidx.i100.3 = getelementptr inbounds <8 x float>, ptr addrspace(1) %out_tiles, i32 %add.i99.3
  store <8 x float> %43, ptr addrspace(1) %arrayidx.i100.3, align 32, !tbaa !14
  %add.i99.4 = add nsw i32 %call, 128
  %arrayidx.i100.4 = getelementptr inbounds <8 x float>, ptr addrspace(1) %out_tiles, i32 %add.i99.4
  store <8 x float> %48, ptr addrspace(1) %arrayidx.i100.4, align 32, !tbaa !14
  %add.i99.5 = add nsw i32 %call, 160
  %arrayidx.i100.5 = getelementptr inbounds <8 x float>, ptr addrspace(1) %out_tiles, i32 %add.i99.5
  store <8 x float> %53, ptr addrspace(1) %arrayidx.i100.5, align 32, !tbaa !14
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local i32 @_Z13get_global_idj(i32 noundef) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare <8 x float> @llvm.riscv.ventus.mma.m16n16k16.row.col.f32.f16.f16.f32(<4 x i32>, <4 x i32>, <8 x float>) #2

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @llvm.riscv.ventus.vgelu.approx.f32(float) #3

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.riscv.ventus.vcvt.fp16.fp32(float) #3

attributes #0 = { convergent nofree norecurse nounwind memory(argmem: readwrite) vscale_range(1,2048) "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="256" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="ventus-gpgpu" "target-features"="+32bit,+a,+m,+relax,+zdinx,+zfinx,+zhinx,+zve32f,+zve32x,+zvl32b,-64bit,-save-restore" "uniform-work-group-size"="false" }
attributes #1 = { convergent mustprogress nofree nounwind willreturn memory(none) "disable-tail-calls"="true" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="ventus-gpgpu" "target-features"="+32bit,+a,+m,+relax,+zdinx,+zfinx,+zhinx,+zve32f,+zve32x,+zvl32b,-64bit,-save-restore" }
attributes #2 = { convergent mustprogress nofree nounwind willreturn memory(none) }
attributes #3 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!opencl.ocl.version = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"ilp32"}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 1, !"SmallDataLimit", i32 8}
!4 = !{i32 2, i32 0}
!5 = !{!"clang version 16.0.0 (https://github.com/THU-DSP-LAB/llvm-project.git 9e00e24e163b41513cc2118e3888d9f87b7715b0)"}
!6 = !{i32 1, i32 1, i32 1, i32 1}
!7 = !{!"none", !"none", !"none", !"none"}
!8 = !{!"uint*", !"uint*", !"uint*", !"float*"}
!9 = !{!"const", !"const", !"const", !""}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
!14 = !{!12, !12, i64 0}
