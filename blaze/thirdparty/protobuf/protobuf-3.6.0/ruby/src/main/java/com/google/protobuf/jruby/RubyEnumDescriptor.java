/*
 * Protocol Buffers - Google's data interchange format
 * Copyright 2014 Google Inc.  All rights reserved.
 * https://developers.google.com/protocol-buffers/
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.google.protobuf.jruby;

import com.google.protobuf.DescriptorProtos;
import com.google.protobuf.Descriptors;
import org.jruby.Ruby;
import org.jruby.RubyClass;
import org.jruby.RubyModule;
import org.jruby.RubyObject;
import org.jruby.RubyNumeric;
import org.jruby.anno.JRubyClass;
import org.jruby.anno.JRubyMethod;
import org.jruby.runtime.Block;
import org.jruby.runtime.ObjectAllocator;
import org.jruby.runtime.ThreadContext;
import org.jruby.runtime.builtin.IRubyObject;

@JRubyClass(name = "EnumDescriptor", include = "Enumerable")
public class RubyEnumDescriptor extends RubyObject {
    public static void createRubyEnumDescriptor(Ruby runtime) {
        RubyModule mProtobuf = runtime.getClassFromPath("Google::Protobuf");
        RubyClass cEnumDescriptor = mProtobuf.defineClassUnder("EnumDescriptor", runtime.getObject(), new ObjectAllocator() {
            @Override
            public IRubyObject allocate(Ruby runtime, RubyClass klazz) {
                return new RubyEnumDescriptor(runtime, klazz);
            }
        });
        cEnumDescriptor.includeModule(runtime.getEnumerable());
        cEnumDescriptor.defineAnnotatedMethods(RubyEnumDescriptor.class);
    }

    public RubyEnumDescriptor(Ruby runtime, RubyClass klazz) {
        super(runtime, klazz);
    }

    /*
     * call-seq:
     *     EnumDescriptor.new => enum_descriptor
     *
     * Creates a new, empty, enum descriptor. Must be added to a pool before the
     * enum type can be used. The enum type may only be modified prior to adding to
     * a pool.
     */
    @JRubyMethod
    public IRubyObject initialize(ThreadContext context) {
        this.builder = DescriptorProtos.EnumDescriptorProto.newBuilder();
        return this;
    }

    /*
     * call-seq:
     *     EnumDescriptor.name => name
     *
     * Returns the name of this enum type.
     */
    @JRubyMethod(name = "name")
    public IRubyObject getName(ThreadContext context) {
        return this.name;
    }

    /*
     * call-seq:
     *     EnumDescriptor.name = name
     *
     * Sets the name of this enum type. Cannot be called if the enum type has
     * already been added to a pool.
     */
    @JRubyMethod(name = "name=")
    public IRubyObject setName(ThreadContext context, IRubyObject name) {
        this.name = name;
        this.builder.setName(Utils.escapeIdentifier(name.asJavaString()));
        return context.runtime.getNil();
    }

    /*
     * call-seq:
     *     EnumDescriptor.add_value(key, value)
     *
     * Adds a new key => value mapping to this enum type. Key must be given as a
     * Ruby symbol. Cannot be called if the enum type has already been added to a
     * pool. Will raise an exception if the key or value is already in use.
     */
    @JRubyMethod(name = "add_value")
    public IRubyObject addValue(ThreadContext context, IRubyObject name, IRubyObject number) {
        DescriptorProtos.EnumValueDescriptorProto.Builder valueBuilder = DescriptorProtos.EnumValueDescriptorProto.newBuilder();
        valueBuilder.setName(name.asJavaString());
        valueBuilder.setNumber(RubyNumeric.num2int(number));
        this.builder.addValue(valueBuilder);
        return context.runtime.getNil();
    }

    /*
     * call-seq:
     *     EnumDescriptor.each(&block)
     *
     * Iterates over key => value mappings in this enum's definition, yielding to
     * the block with (key, value) arguments for each one.
     */
    @JRubyMethod
    public IRubyObject each(ThreadContext context, Block block) {
        Ruby runtime = context.runtime;
        for (Descriptors.EnumValueDescriptor enumValueDescriptor : descriptor.getValues()) {
            block.yield(context, runtime.newArray(runtime.newSymbol(enumValueDescriptor.getName()),
                    runtime.newFixnum(enumValueDescriptor.getNumber())));
        }
        return runtime.getNil();
    }

    /*
     * call-seq:
     *     EnumDescriptor.enummodule => module
     *
     * Returns the Ruby module corresponding to this enum type. Cannot be called
     * until the enum descriptor has been added to a pool.
     */
    @JRubyMethod
    public IRubyObject enummodule(ThreadContext context) {
        if (this.klazz == null) {
            this.klazz = buildModuleFromDescriptor(context);
        }
        return this.klazz;
    }

    public void setDescriptor(Descriptors.EnumDescriptor descriptor) {
        this.descriptor = descriptor;
    }

    public Descriptors.EnumDescriptor getDescriptor() {
        return this.descriptor;
    }

    public DescriptorProtos.EnumDescriptorProto.Builder getBuilder() {
        return this.builder;
    }

    private RubyModule buildModuleFromDescriptor(ThreadContext context) {
        Ruby runtime = context.runtime;
        Utils.checkNameAvailability(context, name.asJavaString());

        RubyModule enumModule = RubyModule.newModule(runtime);
        for (Descriptors.EnumValueDescriptor value : descriptor.getValues()) {
            enumModule.defineConstant(value.getName(), runtime.newFixnum(value.getNumber()));
        }

        enumModule.instance_variable_set(runtime.newString(Utils.DESCRIPTOR_INSTANCE_VAR), this);
        enumModule.defineAnnotatedMethods(RubyEnum.class);
        return enumModule;
    }

    private IRubyObject name;
    private RubyModule klazz;
    private Descriptors.EnumDescriptor descriptor;
    private DescriptorProtos.EnumDescriptorProto.Builder builder;
}
